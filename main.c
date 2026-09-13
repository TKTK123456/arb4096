#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <inttypes.h>
#include <stdatomic.h>
#include <math.h>

#include "generator.h"
#include "biomenoise.h"
#include "filter.h"
#include "gpu_host.h"
#include "tables/btree263.h"

#define START_SEED      1000000000000000000ULL
#define SAMPLE_RANGE    4096
#define SAMPLE_STEP     64
#define TRUE_STEP       4
#define ROUGH_MARGIN    2.0
#define COARSE_SCALE    256
#define PRE_STRIDE      512
#define SAMPLE_Y        256
#define NUM_THREADS     16
#define SCORE_CUTOFF    81.0
#define REQUIRE_ALL_BIOMES 1      // 1 = only keep 52/52 seeds 0 = keep any score over threshold
#define TE_CUTOFF       0.984     // every 83+ seed so far is above this
#define CRUNCH_A        0.25f
#define CRUNCH_AB       0.30f
#define CRUNCH_SUM      0.80f
#define PRE_CUTOFF      60.0     // lowest reduced-pre on an 80+ hit was ~63
#define COARSE_CUTOFF   76.0     // same, from the 80+ results
#define TE_STRIDE       512
#define TE_OCTAVE       2
#define RED_OCTAVE      2
#define MC_VERSION      MC_26_3
#define RESULTS_FILE    "arb4096_results.txt"
#define CHECKPOINT_FILE "arb4096_checkpoint.txt"
#define CKPT_INTERVAL   30
#define STATUS_INTERVAL 20
#define BATCH_SIZE      64
#define USE_GPU         1        // 1 = GPU (falls back to the CPU when there is no usable GPU) 0 = CPU only
#define USE_FILTERS     0        // 0 = crunch() is the first gate 1 = the filters listed in filters/FILTER_RULES
#define FILTER_RULES    "order.txt"

typedef struct { const char *name; int id; double base; } BiomeEntry;
static const BiomeEntry BIOME_TABLE[] = {
#include "biome_table.inc"
};
#define NUM_BIOMES ((int)(sizeof(BIOME_TABLE)/sizeof(BIOME_TABLE[0])))
#define MAX_BIOME_ID 256
#define CRS_W  (SAMPLE_RANGE/COARSE_SCALE + 1)
#define TRUE_W (SAMPLE_RANGE/TRUE_STEP + 1)

static int g_idToIndex[MAX_BIOME_ID];
static void initTables(void){
    for (int i=0;i<MAX_BIOME_ID;i++) g_idToIndex[i]=-1;
    for (int i=0;i<NUM_BIOMES;i++){ int id=BIOME_TABLE[i].id;
        if (id>=0 && id<MAX_BIOME_ID) g_idToIndex[id]=i;
        else fprintf(stderr,"WARNING: biome '%s' id %d out of range\n",BIOME_TABLE[i].name,id); }
}

static inline double arbFromCounts(long long *counts){
    double total=0.0; for(int i=0;i<NUM_BIOMES;i++) total+=(double)counts[i];
    if(total<=0.0) return 0.0;
    double H=0.0; for(int i=0;i<NUM_BIOMES;i++){ if(counts[i]){ double p=(double)counts[i]/total; H+=p*log(p); } }
    H=-H; double normH=H/log((double)NUM_BIOMES); return normH*normH*100.0;
}
static inline int presentCount(long long *counts){ int n=0; for(int i=0;i<NUM_BIOMES;i++) if(counts[i]) n++; return n; }


static inline float oct0(Xoroshiro *px, uint64_t mlo, uint64_t mhi){
    uint64_t lo=xNextLong(px), hi=xNextLong(px);
    Xoroshiro xr={lo^mlo, hi^mhi}; xSkipN(&xr,1);
    return fabsf(((uint32_t)(xNextLong(&xr)>>32)&0xFFFFFF)*5.960464478E-8f - 0.5f);
}
static int crunch(uint64_t seed){
    Xoroshiro xr; xSetSeed(&xr, seed);
    uint64_t lo=xNextLong(&xr), hi=xNextLong(&xr);
    Xoroshiro hn={lo^0x81bb4d22e8dc168e, hi^0xf1c8b4bea16303cd};
    float a=oct0(&hn,0x0ef68ec68504005e,0x48b6bf93a2789640); if(a>CRUNCH_A) return 0;
    float b=oct0(&hn,0x0ef68ec68504005e,0x48b6bf93a2789640); if(a+b>CRUNCH_AB) return 0;
    Xoroshiro en={lo^0xd02491e6058f6fd8, hi^0x4792512c94c17a80};
    float c=oct0(&en,0x082fe255f8be6631,0x4e96119e22dedc81);
    float d=oct0(&en,0x082fe255f8be6631,0x4e96119e22dedc81);
    return a+b+c+d<=CRUNCH_SUM;
}

static const double TEMP_BAND[4] = {-0.45, -0.15, 0.2, 0.55};
static const double TEMP_BAND_BIOMES[5] = {29, 38, 40, 38, 21};    // weightings
static double temp_even(uint64_t seed){
    BiomeNoise bn; setClimateParaSeed(&bn, seed, 0, NP_TEMPERATURE, TE_OCTAVE);
    int cnt[5]={0}, tot=0, half=SAMPLE_RANGE/2;
    for(int z=-half; z<=half; z+=TE_STRIDE)
        for(int x=-half; x<=half; x+=TE_STRIDE){
            double v=sampleClimatePara(&bn,NULL,x/4.0,z/4.0);
            int b=0; while(b<4 && v>=TEMP_BAND[b]) b++;
            cnt[b]++; tot++;
        }
    double H=0.0, Wt=0.0; for(int b=0;b<5;b++) Wt+=TEMP_BAND_BIOMES[b];
    for(int b=0;b<5;b++) if(cnt[b]){ double p=(double)cnt[b]/tot; H += -p*log(p/TEMP_BAND_BIOMES[b]); }
    return H/log(Wt);
}

typedef struct { BiomeNoise t,h,c,e,w; } RedN;
static double prefilter(RedN *rn, uint64_t seed, long long *counts){
    setClimateParaSeed(&rn->t, seed, 0, NP_TEMPERATURE,     RED_OCTAVE);
    setClimateParaSeed(&rn->h, seed, 0, NP_HUMIDITY,        RED_OCTAVE);
    setClimateParaSeed(&rn->c, seed, 0, NP_CONTINENTALNESS, RED_OCTAVE);
    setClimateParaSeed(&rn->e, seed, 0, NP_EROSION,         RED_OCTAVE);
    setClimateParaSeed(&rn->w, seed, 0, NP_WEIRDNESS,       RED_OCTAVE);
    memset(counts,0,sizeof(long long)*NUM_BIOMES);
    int half=SAMPLE_RANGE/2;
    for(int bz=-half; bz<=half; bz+=PRE_STRIDE)
        for(int bx=-half; bx<=half; bx+=PRE_STRIDE){
            double qx=bx/4.0, qz=bz/4.0; uint64_t np[6];
            np[0]=(uint64_t)(int64_t)llround(sampleClimatePara(&rn->t,NULL,qx,qz)*10000.0);
            np[1]=(uint64_t)(int64_t)llround(sampleClimatePara(&rn->h,NULL,qx,qz)*10000.0);
            np[2]=(uint64_t)(int64_t)llround(sampleClimatePara(&rn->c,NULL,qx,qz)*10000.0);
            np[3]=(uint64_t)(int64_t)llround(sampleClimatePara(&rn->e,NULL,qx,qz)*10000.0);
            np[4]=0;
            np[5]=(uint64_t)(int64_t)llround(sampleClimatePara(&rn->w,NULL,qx,qz)*10000.0);
            uint64_t dat=0; int b=climateToBiome(MC_VERSION,np,&dat);
            if(b>=0&&b<MAX_BIOME_ID){int idx=g_idToIndex[b]; if(idx>=0)counts[idx]++;}
        }
    return arbFromCounts(counts);
}
static double coarse(Generator *g, int *buf, long long *counts){
    int W=CRS_W, xb=-(SAMPLE_RANGE/2)/COARSE_SCALE, zb=xb, yb=SAMPLE_Y/4;
    Range r={COARSE_SCALE,xb,zb,W,W,yb,1};
    if (genBiomes(g,buf,r)) return -1e9;
    memset(counts,0,sizeof(long long)*NUM_BIOMES);
    for (int i=0;i<W*W;i++){ int b=buf[i]; if(b>=0&&b<MAX_BIOME_ID){int idx=g_idToIndex[b]; if(idx>=0)counts[idx]++;} }
    return arbFromCounts(counts);
}
static double gridscan(Generator *g, long long *counts){
    memset(counts,0,sizeof(long long)*NUM_BIOMES);
    int half=SAMPLE_RANGE/2, y=SAMPLE_Y/4;
    for(int bz=-half; bz<=half; bz+=SAMPLE_STEP)
        for(int bx=-half; bx<=half; bx+=SAMPLE_STEP){
            int cell; Range r={4, bx/4, bz/4, 1,1, y, 1};
            if (genBiomes(g,&cell,r)) return -1e9;
            int b=cell; if(b>=0&&b<MAX_BIOME_ID){int idx=g_idToIndex[b]; if(idx>=0)counts[idx]++;}
        }
    return arbFromCounts(counts);
}
static double fullscan(Generator *g, int *tbuf, long long *counts){
    int W=TRUE_W, xb=-(SAMPLE_RANGE/2)/4, zb=xb, yb=SAMPLE_Y/4;
    Range r={4,xb,zb,W,W,yb,1};
    if (genBiomes(g,tbuf,r)) return -1e9;
    memset(counts,0,sizeof(long long)*NUM_BIOMES);
    for(int i=0;i<W*W;i++){ int b=tbuf[i]; if(b>=0&&b<MAX_BIOME_ID){int idx=g_idToIndex[b]; if(idx>=0)counts[idx]++;} }
    return arbFromCounts(counts);
}

static volatile sig_atomic_t g_stop=0;
static void onsig(int s){ (void)s; g_stop=1; }
static atomic_uint_fast64_t g_next=0, g_scanned=0, g_hits=0;
static _Atomic double g_best=0.0;
static _Atomic uint64_t g_bestSeed=0;
static pthread_mutex_t g_mx=PTHREAD_MUTEX_INITIALIZER;
static FILE *g_out=NULL;
static uint64_t g_end=0;          // --count: the first seed not to scan
static int g_haveEnd=0, g_useGpu=0;

typedef struct { int *cbuf, *tbuf; long long *counts; } Scr;
static void scr_init(Scr*s){ s->cbuf=malloc(sizeof(int)*CRS_W*CRS_W); s->tbuf=malloc(sizeof(int)*TRUE_W*TRUE_W); s->counts=malloc(sizeof(long long)*NUM_BIOMES); }
static void scr_free(Scr*s){ free(s->cbuf); free(s->tbuf); free(s->counts); }

static Filter g_filter[F_MAXCHAIN];
static int g_nfilter=0;

// The filter files named in filters/FILTER_RULES, one per line, in order.
static int loadFilters(void)
{
    const char *rules = "filters/" FILTER_RULES;
    FILE *f = fopen(rules, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", rules); return -1; }
    char line[512], path[600];
    int bad = 0;
    while (!bad && fgets(line, sizeof(line), f)) {
        char *p = line, *e = line + strlen(line);
        while (e > p && isspace((unsigned char)e[-1])) *--e = 0;
        while (isspace((unsigned char)*p)) p++;
        if (!*p || *p == '#') continue;
        snprintf(path, sizeof(path), "filters/%s", p);
        int got = filterRead(path, g_filter + g_nfilter, F_MAXCHAIN - g_nfilter);
        if (got < 0) { bad = 1; break; }
        for (int k = g_nfilter; k < g_nfilter + got; k++) {
            const Filter *F = &g_filter[k];
            fprintf(stderr, "filter %d: %s (%s)\n", k + 1, F->name, p);
            // a filter made to run after other filters names them in its file
            int same = F->nafter == k;
            for (int j = 0; j < k && same; j++) if (F->after[j] != g_filter[j].hash) same = 0;
            if (!same) fprintf(stderr, "  note: it was made to follow %d filter(s), not the %d before it in %s\n",
                               F->nafter, k, rules);
        }
        g_nfilter += got;
    }
    fclose(f);
    if (!bad && !g_nfilter) { fprintf(stderr, "%s lists no filter files\n", rules); bad = 1; }
    return bad ? -1 : 0;
}

static inline int gatePass(FCtx *x, uint64_t seed){
    if (!USE_FILTERS) return crunch(seed);
    fCtxSeed(x, seed);
    for (int f = 0; f < g_nfilter; f++) if (!filterPass(x, &g_filter[f])) return 0;
    return 1;
}

static void scanSeed(Generator *g, Scr *s, RedN *rn, FCtx *fx, uint64_t seed){
    const double roughGate = SCORE_CUTOFF - ROUGH_MARGIN;
    if (!gatePass(fx, seed)) return;
    if (temp_even(seed) < TE_CUTOFF) return;
    if (prefilter(rn,seed,s->counts) < PRE_CUTOFF) return;
    applySeed(g,DIM_OVERWORLD,seed);
    if (coarse(g,s->cbuf,s->counts) < COARSE_CUTOFF) return;
    double rough = gridscan(g,s->counts);
    if (rough < roughGate) return;
    double arb = fullscan(g,s->tbuf,s->counts);
#if REQUIRE_ALL_BIOMES
    if (presentCount(s->counts) < NUM_BIOMES) return;
#endif
    if (arb > atomic_load(&g_best)){
        pthread_mutex_lock(&g_mx);
        if(arb>atomic_load(&g_best)){ atomic_store(&g_best,arb); atomic_store(&g_bestSeed,seed);} pthread_mutex_unlock(&g_mx);
    }
    if (arb > SCORE_CUTOFF){
        int nb=presentCount(s->counts);
        atomic_fetch_add(&g_hits,1);
        pthread_mutex_lock(&g_mx);
        printf("Seed: %" PRId64 "  ARBITRATIONS: %.6f  biomes: %d/%d\n",(int64_t)seed,arb,nb,NUM_BIOMES); fflush(stdout);
        if(g_out){ fprintf(g_out,"%" PRId64 "\t%.6f\t%d\n",(int64_t)seed,arb,nb); fflush(g_out);} pthread_mutex_unlock(&g_mx);
    }
}

static void *worker(void *a){
    (void)a; Generator g; setupGenerator(&g,MC_VERSION,0); Scr s; scr_init(&s); RedN rn; FCtx fx; fCtxInit(&fx);
    while(!g_stop){
        uint64_t s0=atomic_fetch_add(&g_next,BATCH_SIZE), s1=s0+BATCH_SIZE;
        if (g_haveEnd){ if (s0>=g_end) break; if (s1>g_end) s1=g_end; }
        for(uint64_t seed=s0; seed<s1 && !g_stop; seed++){
            atomic_fetch_add(&g_scanned,1);
            scanSeed(&g,&s,&rn,&fx,seed);
        }
    }
    fCtxFree(&fx); scr_free(&s); return NULL;
}

// ---- GPU
// One thread hands batches of seeds to the GPU; the GPU returns the seeds that
// pass every gate, and the other threads score those again with scanSeed,
// which alone records hits. The checkpoint only moves past a batch once all of
// its returned seeds have been scored, so a stopped run resumes without a gap.

#define GPU_QUEUE_MAX   20000    // seeds waiting for the CPU before the GPU pauses

typedef struct { uint64_t base; int64_t n, pending; } Batch;
typedef struct { uint64_t seed; long batch; } QItem;

static struct {
    pthread_mutex_t mx;
    pthread_cond_t work, room;
    Batch *batch;
    long nbatch, capBatch, firstOpen;
    QItem *q;
    long qhead, qlen, qcap;
    int feederDone, failed;
    uint64_t frontier;          // every seed below this has been scanned
    double gpuBest;             // the GPU's best fullscan value, for the status line
    uint64_t gpuBestSeed;
} Q = { .mx = PTHREAD_MUTEX_INITIALIZER, .work = PTHREAD_COND_INITIALIZER, .room = PTHREAD_COND_INITIALIZER };

static void advanceFrontier(void)
{
    while (Q.firstOpen < Q.nbatch && Q.batch[Q.firstOpen].pending == 0) {
        Q.frontier = Q.batch[Q.firstOpen].base + (uint64_t)Q.batch[Q.firstOpen].n;
        Q.firstOpen++;
    }
}

static double nowSec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *feeder(void *a)
{
    (void)a;
    const int64_t cap = 1 << 20;
    uint64_t *out = malloc(8 * (size_t)cap);
    int64_t n = 1 << 24;
    while (!g_stop && out) {
        uint64_t base = atomic_load(&g_next);
        int64_t m = n;
        if (g_haveEnd) {
            if (base >= g_end) break;
            if ((uint64_t)m > g_end - base) m = (int64_t)(g_end - base);
        }
        pthread_mutex_lock(&Q.mx);
        while (!g_stop && Q.qlen > GPU_QUEUE_MAX) pthread_cond_wait(&Q.room, &Q.mx);
        pthread_mutex_unlock(&Q.mx);
        if (g_stop) break;

        double t0 = nowSec(), best = -1;
        int64_t nout = 0;
        uint64_t bestSeed = 0;
        if (GPU.run(base, m, out, cap, &nout, &best, &bestSeed) != 0 || nout > cap) {
            fprintf(stderr, "GPU batch at %" PRIu64 " failed: %s\n", base,
                    nout > cap ? "more seeds passed than it can return" : GPU.error());
            Q.failed = 1;
            g_stop = 1;
            break;
        }
        double dt = nowSec() - t0;

        pthread_mutex_lock(&Q.mx);
        if (Q.nbatch == Q.capBatch) {
            Q.capBatch = Q.capBatch ? Q.capBatch * 2 : 1024;
            Q.batch = realloc(Q.batch, (size_t)Q.capBatch * sizeof(Batch));
        }
        long bi = Q.nbatch++;
        Q.batch[bi] = (Batch){ base, m, nout };
        for (int64_t i = 0; i < nout; i++) {
            if (Q.qlen == Q.qcap) {
                long nc = Q.qcap ? Q.qcap * 2 : 4096;
                QItem *nq = malloc((size_t)nc * sizeof(QItem));
                for (long k = 0; k < Q.qlen; k++) nq[k] = Q.q[(Q.qhead + k) % Q.qcap];
                free(Q.q);
                Q.q = nq;
                Q.qcap = nc;
                Q.qhead = 0;
            }
            Q.q[(Q.qhead + Q.qlen) % Q.qcap] = (QItem){ out[i], bi };
            Q.qlen++;
        }
        if (best > Q.gpuBest) { Q.gpuBest = best; Q.gpuBestSeed = bestSeed; }
        advanceFrontier();
        pthread_cond_broadcast(&Q.work);
        pthread_mutex_unlock(&Q.mx);
        atomic_store(&g_next, base + (uint64_t)m);
        atomic_fetch_add(&g_scanned, (uint64_t)m);

        // batches of about a second
        if (m == n) {
            if (dt < 0.5 && n < (1LL << 30)) n *= 2;
            else if (dt > 2.0 && n > (1LL << 20)) n /= 2;
        }
    }
    pthread_mutex_lock(&Q.mx);
    Q.feederDone = 1;
    pthread_cond_broadcast(&Q.work);
    pthread_mutex_unlock(&Q.mx);
    free(out);
    return NULL;
}

static void *confirmer(void *a)
{
    (void)a; Generator g; setupGenerator(&g,MC_VERSION,0); Scr s; scr_init(&s); RedN rn; FCtx fx; fCtxInit(&fx);
    for (;;) {
        pthread_mutex_lock(&Q.mx);
        while (Q.qlen == 0 && !Q.feederDone) pthread_cond_wait(&Q.work, &Q.mx);
        if (Q.qlen == 0) { pthread_mutex_unlock(&Q.mx); break; }
        QItem it = Q.q[Q.qhead];
        Q.qhead = (Q.qhead + 1) % Q.qcap;
        Q.qlen--;
        if (Q.qlen <= GPU_QUEUE_MAX) pthread_cond_signal(&Q.room);
        pthread_mutex_unlock(&Q.mx);

        scanSeed(&g, &s, &rn, &fx, it.seed);

        pthread_mutex_lock(&Q.mx);
        Q.batch[it.batch].pending--;
        advanceFrontier();
        pthread_mutex_unlock(&Q.mx);
    }
    fCtxFree(&fx); scr_free(&s); return NULL;
}

#define FAIL(...) do { snprintf(why, whyLen, __VA_ARGS__); return -1; } while (0)

static const int NP_OF_STREAM[6] = { NP_TEMPERATURE, NP_HUMIDITY, NP_CONTINENTALNESS, NP_EROSION, NP_WEIRDNESS, NP_SHIFT };

// setClimateParaSeed with nmax octaves sets up (nmax+1)/2 of half A and the
// rest of half B; nmax <= 0 is every octave
#define OCT_A(nmax) ((nmax) > 0 ? ((nmax) + 1) / 2 : 99)
#define OCT_B(nmax) ((nmax) > 0 ? (nmax) / 2 : 99)

static int addClimate(GfLayout *L, int count, int nA, int nB)
{
    for (int k = 0; k < count; k++)
        if (gpuAddStream(L, NP_OF_STREAM[k], nA, nB) != k) return -1;
    return 0;
}

static int gpuSetup(char *why, size_t whyLen)
{
    static GfSetup S;
    static GfSpline spline[GF_MAXSPLINE];
    memset(&S, 0, sizeof(S));
    if (MC_VERSION != MC_26_3) FAIL("the GPU code only has the biome tree of MC_26_3");
    S.order = btree263_order;
    S.nsteps = (int32_t)(sizeof(btree263_steps) / sizeof(btree263_steps[0]));
    for (int k = 0; k < S.nsteps && k < 16; k++) S.steps[k] = btree263_steps[k];
    S.nparam = (int32_t)(sizeof(btree263_param) / sizeof(btree263_param[0]));
    S.param = &btree263_param[0][0];
    S.len = (int32_t)(sizeof(btree263_nodes) / sizeof(btree263_nodes[0]));
    S.nodes = btree263_nodes;

    BiomeNoise *bn = malloc(sizeof(BiomeNoise));
    if (!bn) FAIL("out of memory");
    initBiomeNoise(bn, MC_VERSION);
    S.nspline = gpuFlattenSpline(bn, spline, GF_MAXSPLINE, &S.splineRoot);
    free(bn);
    if (S.nspline < 0) FAIL("the depth spline does not fit the GPU's table");
    S.spline = spline;

    if (NUM_BIOMES > GF_MAXBIOME) FAIL("more biomes than the GPU code counts");
    for (int i = 0; i < 256; i++) S.idToIndex[i] = (int16_t)g_idToIndex[i];
    S.numBiomes = NUM_BIOMES;
    S.sampleRange = SAMPLE_RANGE;
    S.teStride = TE_STRIDE;
    S.preStride = PRE_STRIDE;
    S.coarseScale = COARSE_SCALE;
    S.sampleY = SAMPLE_Y;
    S.sampleStep = SAMPLE_STEP;
    S.trueStep = TRUE_STEP;
    for (int k = 0; k < 4; k++) S.teBand[k] = TEMP_BAND[k];
    for (int k = 0; k < 5; k++) S.teWeight[k] = TEMP_BAND_BIOMES[k];
    S.teCut = TE_CUTOFF;
    S.preCut = PRE_CUTOFF;
    S.coarseCut = COARSE_CUTOFF;
    S.gridCut = SCORE_CUTOFF - ROUGH_MARGIN;
    S.fullCut = SCORE_CUTOFF;
    S.minBiomes = REQUIRE_ALL_BIOMES ? NUM_BIOMES : 0;
    if (gpuAddStream(&S.te, NP_TEMPERATURE, OCT_A(TE_OCTAVE), OCT_B(TE_OCTAVE)) != 0 ||
        addClimate(&S.pre, 5, OCT_A(RED_OCTAVE), OCT_B(RED_OCTAVE)) ||
        addClimate(&S.coarse, 5, 99, 99) || addClimate(&S.full, 6, 99, 99))
        FAIL("the octave layouts do not fit");
    if (GPU.setup(&S) != 0) FAIL("%s", GPU.error());
    return 0;
}

// The filters as a program for the GPU's first gate.
static int gpuGateSetup(char *why, size_t whyLen)
{
    if (!USE_FILTERS) {
        if (GPU.gate(NULL) != 0) FAIL("%s", GPU.error());
        return 0;
    }
    static GfGate G;
    memset(&G, 0, sizeof(G));
    G.nfilter = g_nfilter;
    for (int pi = 0; pi < F_PARAMS; pi++) {
        G.pmd5[pi][0] = F_PSPEC[pi].md5lo;
        G.pmd5[pi][1] = F_PSPEC[pi].md5hi;
        for (int oc = 0; oc < F_OCT; oc++) {
            int li = f_liveIdx[pi][oc];
            G.omd5[pi][oc][0] = F_MD5_OCT[16 + F_PSPEC[pi].omin + li][0];
            G.omd5[pi][oc][1] = F_MD5_OCT[16 + F_PSPEC[pi].omin + li][1];
        }
    }
    for (int k = 0; k < 8; k++) {
        int nx, nz, fx, fz;
        fFarNearPoint(k, &nx, &nz, &fx, &fz);
        G.farnear[k] = nx; G.farnear[8 + k] = nz; G.farnear[16 + k] = fx; G.farnear[24 + k] = fz;
    }

    // one stream per climate noise any rule samples, with the octaves they read
    int needA[NP_MAX] = {0}, needB[NP_MAX] = {0}, streamOf[NP_MAX];
    for (int f = 0; f < g_nfilter; f++)
        for (int a = 0; a < g_filter[f].natom; a++) {
            const FAtom *A = &g_filter[f].atom[a];
            if (A->kind != FA_RULE) continue;
            for (int np = 0; np < NP_MAX; np++) {
                if (A->rule.needA[np] > needA[np]) needA[np] = A->rule.needA[np];
                if (A->rule.needB[np] > needB[np]) needB[np] = A->rule.needB[np];
            }
        }
    for (int np = 0; np < NP_MAX; np++) {
        streamOf[np] = -1;
        if (!needA[np] && !needB[np]) continue;
        streamOf[np] = gpuAddStream(&G.ruleLayout, np, needA[np], needB[np]);
        if (streamOf[np] < 0 || G.ruleLayout.nslot > GF_LOCALSLOT) FAIL("the filters' rules read more octaves than the GPU gate holds");
    }

    for (int f = 0; f < g_nfilter; f++) {
        const Filter *F = &g_filter[f];
        int nodeBase = G.nnode, map[512];
        if (F->natom > 512 || G.nnode + F->E.n > GF_MAXNODE) FAIL("filter %d is too large for the GPU gate", f + 1);
        for (int a = 0; a < F->natom; a++) {
            const FAtom *A = &F->atom[a];
            map[a] = -1;
            if (A->part) continue;
            if (G.natom >= GF_MAXATOM) FAIL("the filters have more atoms than the GPU gate holds");
            GfAtom *g = &G.atom[G.natom];
            if (A->kind == FA_PHASE) {
                g->kind = GF_A_PHASE;
                g->pi[0] = A->opi; g->half[0] = A->ohalf; g->oct[0] = A->ooct; g->off[0] = A->ooff;
                g->sign[0] = A->sign;
            } else if (A->kind == FA_SUM) {
                g->kind = GF_A_SUM;
                g->nsum = A->nsum;
                for (int k = 0; k < A->nsum; k++) {
                    const FAtom *c = &F->atom[A->sumAtom[k]];
                    g->pi[k] = c->opi; g->half[k] = c->ohalf; g->oct[k] = c->ooct; g->off[k] = c->ooff;
                    g->sign[k] = c->sign;
                    g->mu[k] = A->sumMu[k];
                    g->sd[k] = A->sumSd[k];
                }
            } else {
                if (G.nrule >= GF_MAXRULE) FAIL("the filters have more rules than the GPU gate holds");
                const FRule *R = &A->rule;
                GfRule *q = &G.rule[G.nrule];
                q->sym = R->sym; q->agg = R->agg; q->nterm = R->nterm;
                for (int i = 0; i < R->nterm; i++) {
                    int sampled = R->kind[i] == FT_BLOCK || R->kind[i] == FT_FARNEAR;
                    q->kind[i] = R->kind[i];
                    q->metric[i] = R->metric[i];
                    q->bx[i] = R->bx[i];
                    q->bz[i] = R->bz[i];
                    q->sign[i] = R->sign[i];
                    q->mu[i] = R->mu[i];
                    q->sd[i] = R->sd[i];
                    q->pi[i] = fOrigParamIndex(R->termParam[i]);
                    q->stream[i] = sampled ? streamOf[R->termParam[i]] : 0;
                    if (q->stream[i] < 0) FAIL("filter %d has a rule term the GPU gate cannot read", f + 1);
                }
                g->kind = GF_A_RULE;
                g->rule = G.nrule++;
                g->rsign = A->sign;
            }
            map[a] = G.natom++;
        }
        for (int i = 0; i < F->E.n; i++) {
            const FNode *s = &F->E.nd[i];
            GfNode *d = &G.node[nodeBase + i];
            d->type = s->type;
            d->nkid = s->nkid;
            for (int k = 0; k < s->nkid; k++) d->kid[k] = nodeBase + s->kid[k];
            d->atom = s->type == FE_LEAF ? map[s->atom] : 0;
            d->t = s->t;
            if (d->atom < 0) FAIL("filter %d reads an atom the GPU gate does not have", f + 1);
        }
        G.root[f] = nodeBase;
        G.nnode += F->E.n;
    }
    if (GPU.gate(&G) != 0) FAIL("%s", GPU.error());
    return 0;
}

#undef FAIL

static uint64_t progressMark(void){
    uint64_t v;
    if (g_useGpu){ pthread_mutex_lock(&Q.mx); v=Q.frontier; pthread_mutex_unlock(&Q.mx); }
    else v=atomic_load(&g_next);
    return (g_haveEnd && v>g_end) ? g_end : v;
}
static void bestSoFar(double *best, uint64_t *seed){
    *best=atomic_load(&g_best); *seed=atomic_load(&g_bestSeed);
    if (g_useGpu){ pthread_mutex_lock(&Q.mx); if(Q.gpuBest>*best){ *best=Q.gpuBest; *seed=Q.gpuBestSeed; } pthread_mutex_unlock(&Q.mx); }
}

static void writeCheckpoint(void){ uint64_t v=progressMark(); FILE*f=fopen(CHECKPOINT_FILE,"w"); if(f){fprintf(f,"%" PRIu64 "\n",v);fclose(f);} }
static void *ckpt(void*a){(void)a;while(!g_stop){sleep(CKPT_INTERVAL);if(!g_stop)writeCheckpoint();}return NULL;}
static void *stat_t(void*a){(void)a;uint64_t l=atomic_load(&g_scanned);time_t lt=time(NULL);
    while(!g_stop){sleep(STATUS_INTERVAL);uint64_t n=atomic_load(&g_scanned);time_t nt=time(NULL);double dt=difftime(nt,lt);if(dt<=0)dt=1;
        double best; uint64_t bs; bestSoFar(&best,&bs);
        fprintf(stderr,"[status] %.2fM  %.0f/s | hits=%" PRIu64 " | best ARB=%.3f seed %" PRId64 "\n",
            n/1e6,(n-l)/dt,atomic_load(&g_hits),best,(int64_t)bs);
        l=n;lt=nt;}
    return NULL;}

static int runSingle(const char*ss){
    char*e; long long p=strtoll(ss,&e,10); if(e==ss||*e){fprintf(stderr,"bad seed\n");return 1;}
    uint64_t seed=(uint64_t)p;
    double te=temp_even(seed); int cr=crunch(seed);
    Generator g; setupGenerator(&g,MC_VERSION,0); Scr s; scr_init(&s);
    RedN rn; double pre=prefilter(&rn,seed,s.counts);
    applySeed(&g,DIM_OVERWORLD,seed);
    double c=coarse(&g,s.cbuf,s.counts);
    double rough=gridscan(&g,s.counts);
    double f=fullscan(&g,s.tbuf,s.counts);
    int nb=presentCount(s.counts);
    printf("Seed: %lld  ARBITRATIONS: %.6f  biomes: %d/%d   (crunch: %d  temp-even: %.3f  pre: %.3f  gate: %.3f  rough: %.3f)\n",
        (long long)p,f,nb,NUM_BIOMES,cr,te,pre,c,rough);
    if (USE_FILTERS){ FCtx fx; fCtxInit(&fx); printf("  filters: %s\n", gatePass(&fx,seed) ? "pass" : "fail"); fCtxFree(&fx); }
    scr_free(&s); return 0;
}

static int parseU64(const char *s, uint64_t *v){ char *e; *v=strtoull(s,&e,10); return e!=s && !*e; }

int main(int argc,char**argv){
    initTables();
    fInitLiveIdx();
    if (USE_FILTERS && loadFilters()!=0) return 1;
    if (argc>=2 && !strcmp(argv[1],"single")){ if(argc!=3){fprintf(stderr,"usage: %s single <seed>\n",argv[0]);return 1;} return runSingle(argv[2]); }
    uint64_t start=START_SEED, count=0; int haveStart=0;
    for (int i=1;i<argc;i++){
        if (!strcmp(argv[i],"--start") && i+1<argc && parseU64(argv[i+1],&start)){ haveStart=1; i++; }
        else if (!strcmp(argv[i],"--count") && i+1<argc && parseU64(argv[i+1],&count)) i++;
        else { fprintf(stderr,"usage: %s [--start <seed>] [--count <n>] | %s single <seed>\n",argv[0],argv[0]); return 1; }
    }
    if (!haveStart){ FILE*cf=fopen(CHECKPOINT_FILE,"r");
        if(cf){uint64_t v;if(fscanf(cf,"%" SCNu64,&v)==1&&v>start)start=v;fclose(cf);} }
    if (count){ g_end=start+count; g_haveEnd=1; }

    char gpuName[512]="", why[512]="";
    if (USE_GPU){
        if (gpuOpen(0,gpuName,sizeof(gpuName))!=0) snprintf(why,sizeof(why),"%s",gpuName);
        else if (gpuSetup(why,sizeof(why))==0 && gpuGateSetup(why,sizeof(why))==0) g_useGpu=1;
        if (!g_useGpu) fprintf(stderr,"GPU not used: %s - running on the CPU\n",why);
    }

    signal(SIGINT,onsig);
    atomic_store(&g_next,start); Q.frontier=start; g_out=fopen(RESULTS_FILE,"a");
    fprintf(stderr,"biome arbitrations finder | start %" PRIu64 " | %d threads | cutoff %.2f\n",start,NUM_THREADS,(double)SCORE_CUTOFF);
    fprintf(stderr,"  %s%s | first gate: %s\n",g_useGpu?"GPU: ":"CPU",g_useGpu?gpuName:"",USE_FILTERS?"filters/" FILTER_RULES:"crunch()");
    double t0=nowSec();
    pthread_t th[NUM_THREADS],c,s,fd;
    pthread_create(&c,NULL,ckpt,NULL); pthread_create(&s,NULL,stat_t,NULL);
    if (g_useGpu){
        pthread_create(&fd,NULL,feeder,NULL);
        for(int i=0;i<NUM_THREADS;i++)pthread_create(&th[i],NULL,confirmer,NULL);
        pthread_join(fd,NULL);
        for(int i=0;i<NUM_THREADS;i++)pthread_join(th[i],NULL);
    } else {
        for(int i=0;i<NUM_THREADS;i++)pthread_create(&th[i],NULL,worker,NULL);
        for(int i=0;i<NUM_THREADS;i++)pthread_join(th[i],NULL);
    }
    writeCheckpoint();
    double dt=nowSec()-t0, best; uint64_t bs; bestSoFar(&best,&bs);
    fprintf(stderr,"stopped at %" PRIu64 " | %.2fM seeds in %.0fs | hits=%" PRIu64 " | best ARB=%.3f seed %" PRId64 "\n",
        progressMark(),atomic_load(&g_scanned)/1e6,dt,atomic_load(&g_hits),best,(int64_t)bs);
    return Q.failed?1:0;
}
