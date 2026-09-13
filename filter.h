// filter.h - custom filters for the seed finder: filter files as
// a seperate rule finding tool writes them.
//
// Everything that decides whether a seed passes is taken from
// that rule finding tool operation for operation: lazy
// lattice-offset draws, partial octave set-ups, rule terms, sums with
// early exit - so a filter passes exactly the seeds it passed in the 
// rule finding tool. If that tool's evaluation changes, this copy has to follow it.
//
// A filter file holds one or more filters; several files apply in the order
// given, each to what the ones before kept.

#ifndef FILTER_H
#define FILTER_H

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "biomenoise.h"
#include "noise.h"
#include "rng.h"

#define F_WINDOW    4096
#define F_RMAX      (F_WINDOW/2)
#define F_DP        (337.0 / 331.0)
#define F_NAMELEN   48
#define F_MAXTERM   16
#define F_MAXEN     32
#define F_MAXKID    4
#define F_MAXSUMK   4
#define F_MAXCHAIN  8
#define F_LINELEN   16384
#define F_DEFTEXT   (1 << 17)
#define F_HEAD      "hbd-filter 1"

enum { FM_OCT0A, FM_OCT0B, FM_OCT0, FM_OCT01, FM_FULL, FM_NMETRIC };
enum { FS_NONE, FS_MIRROR4, FS_DIHEDRAL8, FS_RING };
enum { FG_ABSMAX, FG_MAX, FG_MIN };

static const char *F_MET_TAG[FM_NMETRIC] = { "oct0a", "oct0b", "oct0", "oct0_1", "full" };
static const char *F_SYM_TAG[4] = { "none", "mirror4", "dihedral8", "ring" };
static const char *F_AGG_TAG[3] = { "absmax", "max", "min" };

#define F_LOG(...) fprintf(stderr, __VA_ARGS__)

static const char *fParamName(int np)
{
    switch (np) {
        case NP_TEMPERATURE:     return "temperature";
        case NP_HUMIDITY:        return "humidity";
        case NP_CONTINENTALNESS: return "continentalness";
        case NP_EROSION:         return "erosion";
        case NP_WEIRDNESS:       return "weirdness";
        default:                 return "unknown";
    }
}
static int fParamFromName(const char *s)
{
    if (!strcmp(s, "temperature"))     return NP_TEMPERATURE;
    if (!strcmp(s, "humidity"))        return NP_HUMIDITY;
    if (!strcmp(s, "continentalness")) return NP_CONTINENTALNESS;
    if (!strcmp(s, "erosion"))         return NP_EROSION;
    if (!strcmp(s, "weirdness"))       return NP_WEIRDNESS;
    return -1;
}
static int fTagIndex(const char *s, const char **t, int n)
{
    for (int i = 0; i < n; i++) if (!strcmp(s, t[i])) return i;
    return -1;
}
static uint64_t fFnv64(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
    return h ? h : 1;
}

// ------------------------------------------------- lattice offset tables

static const uint64_t F_MD5_OCT[17][2] = {
    {0xc613bf766619f992, 0x954753f86691b86a}, // octave_-16
    {0x7eee475a921c6cf5, 0xf2bd39426f8da413},
    {0xfc0027cef9683114, 0xb758d3954dcbfdd3},
    {0xd1fc8a05be565eca, 0xdc2a3915cbdda25b},
    {0xb198de63a8012672, 0x7b84cad43ef7b5a8},
    {0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8},
    {0x36d326eed40efeb2, 0x5be9ce18223c636a},
    {0x082fe255f8be6631, 0x4e96119e22dedc81},
    {0x0ef68ec68504005e, 0x48b6bf93a2789640},
    {0xf11268128982754f, 0x257a1d670430b0aa},
    {0xe51c98ce7d1de664, 0x5f9478a733040c45},
    {0x6d7b49e7e429850a, 0x2e3063c622a24777},
    {0xbd90d5377ba1b762, 0xc07317d419a7548d},
    {0x53d39c6752dac858, 0xbcd1c5a80ab65b3e},
    {0xb4a24d7a84e7677b, 0x023ff9668e89b5c4},
    {0xdffa22b534c5f608, 0xb9b67517d3665ca9},
    {0xd50708086cef4d7c, 0x6e1651ecc7f43309}, // octave_0
};

static const double F_AMP_TEMP[]  = {1.5, 0, 1, 0, 0, 0};
static const double F_AMP_HUM[]   = {1, 1, 0, 0, 0, 0};
static const double F_AMP_CONT[]  = {1, 1, 2, 2, 2, 1, 1, 1, 1};
static const double F_AMP_EROS[]  = {1, 1, 0, 1, 1};
static const double F_AMP_WEIRD[] = {1, 2, 1, 0, 0, 0};

typedef struct {
    int np;
    uint64_t md5lo, md5hi;
    int omin, len;
    const double *amp;
} FPSpec;

#define F_PARAMS 5
#define F_OCT    2

static const FPSpec F_PSPEC[F_PARAMS] = {
    { NP_TEMPERATURE,     0x5c7e6b29735f0d7f, 0xf7d86f1bbc734988, -10, 6, F_AMP_TEMP  },
    { NP_HUMIDITY,        0x81bb4d22e8dc168e, 0xf1c8b4bea16303cd,  -8, 6, F_AMP_HUM   },
    { NP_CONTINENTALNESS, 0x83886c9d0ae3a662, 0xafa638a61b42e8ad,  -9, 9, F_AMP_CONT  },
    { NP_EROSION,         0xd02491e6058f6fd8, 0x4792512c94c17a80,  -9, 5, F_AMP_EROS  },
    { NP_WEIRDNESS,       0xefc8ef4d36102b34, 0x1beeeb324a0f24ea,  -7, 6, F_AMP_WEIRD },
};

static int f_liveIdx[F_PARAMS][F_OCT];

static void fInitLiveIdx(void)
{
    for (int pi = 0; pi < F_PARAMS; pi++) {
        int n = 0;
        for (int i = 0; i < F_PSPEC[pi].len && n < F_OCT; i++)
            if (F_PSPEC[pi].amp[i] != 0) f_liveIdx[pi][n++] = i;
    }
}

static int fOrigParamIndex(int np)
{
    for (int i = 0; i < F_PARAMS; i++) if (F_PSPEC[i].np == np) return i;
    return -1;
}

static void fOffsetName(int pi, int half, int oct, int off, char *buf, size_t n)
{
    snprintf(buf, n, "%s.%c%d.%c", fParamName(F_PSPEC[pi].np), half ? 'B' : 'A', oct, "abc"[off]);
}

static int fParseOffsetName(const char *nm, int *pi, int *half, int *oct, int *off)
{
    char buf[F_NAMELEN];
    snprintf(buf, sizeof(buf), "%s", nm);
    char *d1 = strchr(buf, '.');
    if (!d1) return -1;
    *d1 = 0;
    char *d2 = strchr(d1 + 1, '.');
    if (!d2) return -1;
    *d2 = 0;
    if (strlen(d1 + 1) != 2) return -1;
    int np = fParamFromName(buf);
    if (np < 0) return -1;
    *pi = fOrigParamIndex(np);
    if (*pi < 0) return -1;
    if (d1[1] != 'A' && d1[1] != 'B') return -1;
    *half = (d1[1] == 'B');
    *oct = d1[2] - '0';
    if (*oct < 0 || *oct >= F_OCT) return -1;
    const char *o = d2[1] ? strchr("abc", d2[1]) : NULL;
    if (!o || d2[2]) return -1;
    *off = (int)(o - "abc");
    return 0;
}

static inline double fPhaseOf(double raw) { return fabs((raw - floor(raw)) - 0.5); }
static inline double fOriginOf(double raw) { return fabs(raw / 256.0 - 0.5); }

// ------------------------------------------------------ noise evaluation

typedef struct { const PerlinNoise *A, *B; int nA, nB; double amp; } FParamOct;

static inline double fOctValue(const PerlinNoise *p, double x, double z)
{
    double lf = p->lacunarity;
    return p->amplitude * samplePerlin(p, x * lf, 0.0, z * lf, 0.0, 0.0);
}

static inline void fParamOct(const BiomeNoise *bn, int np, FParamOct *po)
{
    const DoublePerlinNoise *dp = bn->climate + np;
    po->A = dp->octA.octaves; po->nA = dp->octA.octcnt;
    po->B = dp->octB.octaves; po->nB = dp->octB.octcnt;
    po->amp = dp->amplitude;
}

static void fMetricNeed(int metric, int *needA, int *needB)
{
    switch (metric) {
        case FM_OCT0A: *needA = 1;  *needB = 0;  break;
        case FM_OCT0B: *needA = 0;  *needB = 1;  break;
        case FM_OCT0:  *needA = 1;  *needB = 1;  break;
        case FM_OCT01: *needA = 2;  *needB = 2;  break;
        default:       *needA = 99; *needB = 99; break;
    }
}

static inline double fMetricOne(const FParamOct *po, double bx, double bz, int metric)
{
    double qx = bx * 0.25, qz = bz * 0.25;
    double xb = qx * F_DP, zb = qz * F_DP;
    switch (metric) {
    case FM_OCT0A:
        return fOctValue(po->A, qx, qz) * po->amp;
    case FM_OCT0B:
        return fOctValue(po->B, xb, zb) * po->amp;
    case FM_OCT0:
        return (fOctValue(po->A, qx, qz) + fOctValue(po->B, xb, zb)) * po->amp;
    case FM_OCT01: {
        double a = fOctValue(po->A, qx, qz);
        a += fOctValue(po->A + 1, qx, qz);
        double b = fOctValue(po->B, xb, zb);
        b += fOctValue(po->B + 1, xb, zb);
        return (a + b) * po->amp;
    }
    default: {
        double a = 0, b = 0;
        for (int i = 0; i < po->nA; i++) a += fOctValue(po->A + i, qx, qz);
        for (int i = 0; i < po->nB; i++) b += fOctValue(po->B + i, xb, zb);
        return (a + b) * po->amp;
    }
    }
}

static inline double fAggInit(int agg)
{
    if (agg == FG_MAX) return -INFINITY;
    if (agg == FG_MIN) return  INFINITY;
    return 0.0;
}
static inline void fAggUpdate(int agg, double *acc, double v)
{
    if (agg == FG_ABSMAX) { double a = fabs(v); if (a > *acc) *acc = a; }
    else if (agg == FG_MAX) { if (v > *acc) *acc = v; }
    else { if (v < *acc) *acc = v; }
}

static inline int fOrbitPoints(int sym, int bx, int bz, int *ox, int *oz)
{
    int x = bx, z = bz;
    if (sym == FS_NONE) { ox[0] = x; oz[0] = z; return 1; }
    if (x < 0) x = -x;
    if (z < 0) z = -z;
    ox[0]= x; oz[0]= z;  ox[1]= x; oz[1]=-z;
    ox[2]=-x; oz[2]=-z;  ox[3]=-x; oz[3]= z;
    if (sym != FS_DIHEDRAL8) return 4;
    ox[4]= z; oz[4]= x;  ox[5]= z; oz[5]=-x;
    ox[6]=-z; oz[6]=-x;  ox[7]=-z; oz[7]= x;
    return 8;
}

static inline double fOrbitM(const FParamOct *po, int sym, int agg, int metric, int bx, int bz)
{
    int ox[8], oz[8];
    int no = fOrbitPoints(sym, bx, bz, ox, oz);
    double e = fAggInit(agg);
    for (int k = 0; k < no; k++) fAggUpdate(agg, &e, fMetricOne(po, ox[k], oz[k], metric));
    return e;
}

// The k-th (of 8) near and far sample points of a far-minus-near term.
static void fFarNearPoint(int k, int *nx, int *nz, int *fx, int *fz)
{
    double th = k * (2 * M_PI / 8);
    *nx = (int)(0.15 * F_RMAX * cos(th)); *nz = (int)(0.15 * F_RMAX * sin(th));
    *fx = (int)(0.90 * F_RMAX * cos(th)); *fz = (int)(0.90 * F_RMAX * sin(th));
    if (!*nx) *nx = 1;
    if (!*nz) *nz = 1;
    if (!*fx) *fx = 1;
    if (!*fz) *fz = 1;
}

// ------------------------------------------------------------------- rule

enum { FT_BLOCK, FT_FARNEAR, FT_ORIGIN, FT_PHASE };

typedef struct {
    int param, sym, agg;
    int nterm;
    int kind[F_MAXTERM];
    int metric[F_MAXTERM];              // block/farnear: metric; origin/phase: offset
    int bx[F_MAXTERM], bz[F_MAXTERM];   // origin/phase: half, octave
    int termParam[F_MAXTERM];
    double sign[F_MAXTERM];
    double mu[F_MAXTERM], sd[F_MAXTERM];
    double threshold, targetRecall;
    int needFull;
    int needA[NP_MAX], needB[NP_MAX];
} FRule;

static int fParseSeries(const char *name, int fallbackParam, int *param, int *metric)
{
    char buf[F_NAMELEN];
    snprintf(buf, sizeof(buf), "%s", name);
    *param = fallbackParam;
    int m = fTagIndex(buf, F_MET_TAG, FM_NMETRIC);
    if (m >= 0) { *metric = m; return 0; }
    char *us = strrchr(buf, '_');
    while (us) {
        *us = 0;
        int p = fParamFromName(buf);
        int mm = fTagIndex(us + 1, F_MET_TAG, FM_NMETRIC);
        if (p >= 0 && mm >= 0) { *param = p; *metric = mm; return 0; }
        *us = '_';
        char *prev = us - 1;
        while (prev > buf && *prev != '_') prev--;
        us = (prev > buf) ? prev : NULL;
    }
    return -1;
}

static void fRuleDefaults(FRule *R, int defParam)
{
    memset(R, 0, sizeof(*R));
    R->param = defParam; R->sym = FS_MIRROR4; R->agg = FG_ABSMAX; R->targetRecall = 1.0;
}

static int fRuleLine(FRule *R, const char *line, const char *where)
{
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\r' || !*p) return 0;
    char a[64], b[64];
    if (sscanf(p, "param %63s", a) == 1)          { int q = fParamFromName(a); if (q >= 0) R->param = q; }
    else if (sscanf(p, "sym %63s", a) == 1)       { R->sym = fTagIndex(a, F_SYM_TAG, 4); }
    else if (sscanf(p, "agg %63s", a) == 1)       { R->agg = fTagIndex(a, F_AGG_TAG, 3); }
    else if (sscanf(p, "threshold %lf", &R->threshold) == 1) { }
    else if (sscanf(p, "target_recall %lf", &R->targetRecall) == 1) { }
    else if (!strncmp(p, "term ", 5)) {
        if (R->nterm >= F_MAXTERM) { F_LOG("too many terms in %s\n", where); return -1; }
        int i = R->nterm;
        if (sscanf(p, "term %63s %63s", a, b) < 2) { F_LOG("bad term line in %s\n", where); return -1; }
        if (!strcmp(a, "origin") || !strcmp(a, "phase")) {
            char nm[F_NAMELEN];
            if (sscanf(p, "%*s %*s %47s %lf %lf %lf", nm, &R->sign[i], &R->mu[i], &R->sd[i]) != 4) {
                F_LOG("bad %s term in %s\n", a, where); return -1; }
            int pi, half, oct, off;
            if (fParseOffsetName(nm, &pi, &half, &oct, &off) != 0) {
                F_LOG("unparseable offset name '%s' in %s\n", nm, where); return -1; }
            R->kind[i] = !strcmp(a, "origin") ? FT_ORIGIN : FT_PHASE;
            R->termParam[i] = F_PSPEC[pi].np;
            R->bx[i] = half; R->bz[i] = oct; R->metric[i] = off;
        } else {
            int pp, mm;
            if (fParseSeries(a, R->param, &pp, &mm) != 0) {
                F_LOG("unknown series '%s' in %s\n", a, where); return -1; }
            R->termParam[i] = pp;
            R->metric[i] = mm;
            if (R->param < 0) R->param = pp;
            if (!strcmp(b, "farnear")) {
                R->kind[i] = FT_FARNEAR;
                if (sscanf(p, "%*s %*s %*s %lf %lf %lf", &R->sign[i], &R->mu[i], &R->sd[i]) != 3) {
                    F_LOG("bad farnear term in %s\n", where); return -1; }
            } else {
                R->kind[i] = FT_BLOCK;
                if (sscanf(p, "%*s %*s %*s %d %d %lf %lf %lf", &R->bx[i], &R->bz[i],
                           &R->sign[i], &R->mu[i], &R->sd[i]) != 5) {
                    F_LOG("bad block term in %s\n", where); return -1; }
            }
        }
        R->nterm++;
    }
    else return 1;
    return 0;
}

static int fRuleFinish(FRule *R, const char *path)
{
    if (R->sym < 0 || R->agg < 0 || !R->nterm) {
        F_LOG("%s is missing sym/agg or has no terms\n", path);
        return -1;
    }
    for (int i = 0; i < R->nterm; i++) {
        if (R->termParam[i] < 0) { F_LOG("%s has a term with no parameter\n", path); return -1; }
        if (R->kind[i] == FT_BLOCK || R->kind[i] == FT_FARNEAR) {
            int a, b;
            fMetricNeed(R->metric[i], &a, &b);
            int np = R->termParam[i];
            if (a > R->needA[np]) R->needA[np] = a;
            if (b > R->needB[np]) R->needB[np] = b;
            if (R->metric[i] == FM_FULL) R->needFull = 1;
        }
    }
    return 0;
}

static size_t fRuleText(const FRule *R, const char *indent, char *buf, size_t bn)
{
    size_t at = 0;
    #define F_RPUT(...) do { int w_ = snprintf(buf + at, at < bn ? bn - at : 0, __VA_ARGS__); \
        if (w_ > 0) at += (size_t)w_; } while (0)
    F_RPUT("%sparam %s\n%ssym %s\n%sagg %s\n", indent, fParamName(R->param), indent, F_SYM_TAG[R->sym],
           indent, F_AGG_TAG[R->agg]);
    for (int i = 0; i < R->nterm; i++) {
        int np = R->termParam[i];
        if (R->kind[i] == FT_ORIGIN || R->kind[i] == FT_PHASE) {
            char nm[F_NAMELEN];
            fOffsetName(fOrigParamIndex(np), R->bx[i], R->bz[i], R->metric[i], nm, sizeof(nm));
            F_RPUT("%sterm %s %s %.17g %.17g %.17g\n", indent, R->kind[i] == FT_ORIGIN ? "origin" : "phase",
                   nm, R->sign[i], R->mu[i], R->sd[i]);
        } else if (R->kind[i] == FT_FARNEAR) {
            F_RPUT("%sterm %s_%s farnear %.17g %.17g %.17g\n", indent, fParamName(np), F_MET_TAG[R->metric[i]],
                   R->sign[i], R->mu[i], R->sd[i]);
        } else {
            F_RPUT("%sterm %s_%s block %d %d %.17g %.17g %.17g\n", indent, fParamName(np), F_MET_TAG[R->metric[i]],
                   R->bx[i], R->bz[i], R->sign[i], R->mu[i], R->sd[i]);
        }
    }
    #undef F_RPUT
    return at;
}

// -------------------------------------------------------------- atoms

enum { FA_PHASE, FA_SUM, FA_RULE };

typedef struct {
    char name[160];
    int kind;
    double sign;
    int opi, ohalf, ooct, ooff;             // FA_PHASE
    int nsum;                               // FA_SUM
    int sumAtom[F_MAXSUMK];
    double sumMu[F_MAXSUMK], sumSd[F_MAXSUMK];
    double remMin[F_MAXSUMK], remMax[F_MAXSUMK];
    FRule rule;                             // FA_RULE
    int part;                               // a phase read only as a sum's component
} FAtom;

// One per thread: what has been drawn or initialised for the current seed.
typedef struct {
    BiomeNoise *bn[NP_MAX];
    uint64_t seed;
    int haveSeed;
    uint8_t octA[NP_MAX], octB[NP_MAX];
    uint8_t haveBase;
    uint8_t nStream[F_PARAMS];
    uint8_t ndraw[F_PARAMS][2][F_OCT];
    uint64_t xlo, xhi;
    uint64_t slo[F_PARAMS][2], shi[F_PARAMS][2];
    Xoroshiro sub[F_PARAMS][2][F_OCT];
    double raw[F_PARAMS][2][F_OCT][3];
} FCtx;

static int fCtxInit(FCtx *x)
{
    memset(x, 0, sizeof(*x));
    for (int i = 0; i < NP_MAX; i++) {
        x->bn[i] = malloc(sizeof(BiomeNoise));
        if (!x->bn[i]) return -1;
    }
    return 0;
}
static void fCtxFree(FCtx *x)
{
    for (int i = 0; i < NP_MAX; i++) free(x->bn[i]);
}
static inline void fCtxSeed(FCtx *x, uint64_t seed)
{
    if (x->haveSeed && x->seed == seed) return;
    x->seed = seed;
    x->haveSeed = 1;
    memset(x->octA, 0, sizeof(x->octA));
    memset(x->octB, 0, sizeof(x->octB));
    x->haveBase = 0;
    memset(x->nStream, 0, sizeof(x->nStream));
    memset(x->ndraw, 0, sizeof(x->ndraw));
}

static inline double fOffsetRaw(FCtx *x, int pi, int half, int oct, int off)
{
    uint8_t *nd = &x->ndraw[pi][half][oct];
    if (*nd > off) return x->raw[pi][half][oct][off];
    if (!x->haveBase) {
        Xoroshiro r;
        xSetSeed(&r, x->seed);
        x->xlo = xNextLong(&r);
        x->xhi = xNextLong(&r);
        x->haveBase = 1;
    }
    if (x->nStream[pi] <= half) {
        Xoroshiro r;
        r.lo = x->xlo ^ F_PSPEC[pi].md5lo;
        r.hi = x->xhi ^ F_PSPEC[pi].md5hi;
        for (int h = 0; h <= half; h++) {
            x->slo[pi][h] = xNextLong(&r);
            x->shi[pi][h] = xNextLong(&r);
        }
        x->nStream[pi] = (uint8_t)(half + 1);
    }
    Xoroshiro *q = &x->sub[pi][half][oct];
    if (*nd == 0) {
        int li = f_liveIdx[pi][oct];
        q->lo = x->slo[pi][half] ^ F_MD5_OCT[16 + F_PSPEC[pi].omin + li][0];
        q->hi = x->shi[pi][half] ^ F_MD5_OCT[16 + F_PSPEC[pi].omin + li][1];
    }
    while (*nd <= off) {
        x->raw[pi][half][oct][*nd] = xNextDouble(q) * 256.0;
        (*nd)++;
    }
    return x->raw[pi][half][oct][off];
}

static inline const BiomeNoise *fCtxEnsure(FCtx *x, int np, int needA, int needB)
{
    if ((x->octA[np] || x->octB[np]) && x->octA[np] >= needA && x->octB[np] >= needB)
        return x->bn[np];
    int nmax;
    if (needA >= 99 || needB >= 99) nmax = -1;
    else {
        int na = needA > x->octA[np] ? needA : x->octA[np];
        int nb = needB > x->octB[np] ? needB : x->octB[np];
        nmax = 2 * na - 1;
        if (2 * nb > nmax) nmax = 2 * nb;
        if (nmax < 1) nmax = 1;
    }
    setClimateParaSeed(x->bn[np], x->seed, 0, np, nmax);
    const DoublePerlinNoise *dp = x->bn[np]->climate + np;
    x->octA[np] = (uint8_t)(nmax < 0 ? 99 : dp->octA.octcnt);
    x->octB[np] = (uint8_t)(nmax < 0 ? 99 : dp->octB.octcnt);
    return x->bn[np];
}

static inline double fRuleValue(FCtx *x, const FRule *R)
{
    double s = 0;
    for (int i = 0; i < R->nterm; i++) {
        int np = R->termParam[i];
        double v;
        if (R->kind[i] == FT_ORIGIN || R->kind[i] == FT_PHASE) {
            double raw = fOffsetRaw(x, fOrigParamIndex(np), R->bx[i], R->bz[i], R->metric[i]);
            v = R->kind[i] == FT_ORIGIN ? fOriginOf(raw) : fPhaseOf(raw);
        } else {
            const BiomeNoise *bn = fCtxEnsure(x, np, R->needA[np], R->needB[np]);
            FParamOct po;
            fParamOct(bn, np, &po);
            if (R->kind[i] == FT_FARNEAR) {
                double nearSum = 0, farSum = 0;
                for (int k = 0; k < 8; k++) {
                    int nx, nz, fx, fz;
                    fFarNearPoint(k, &nx, &nz, &fx, &fz);
                    nearSum += fOrbitM(&po, R->sym, R->agg, R->metric[i], nx, nz);
                    farSum += fOrbitM(&po, R->sym, R->agg, R->metric[i], fx, fz);
                }
                v = farSum / 8 - nearSum / 8;
            } else {
                v = fOrbitM(&po, R->sym, R->agg, R->metric[i], R->bx[i], R->bz[i]);
            }
        }
        s += R->sign[i] * (v - R->mu[i]) / R->sd[i];
    }
    return s;
}

static inline double fAtomValue(FCtx *x, const FAtom *tab, int ai)
{
    const FAtom *a = &tab[ai];
    switch (a->kind) {
    case FA_PHASE:
        return a->sign * fPhaseOf(fOffsetRaw(x, a->opi, a->ohalf, a->ooct, a->ooff));
    case FA_SUM: {
        double s = 0;
        for (int k = 0; k < a->nsum; k++)
            s += (fAtomValue(x, tab, a->sumAtom[k]) - a->sumMu[k]) / a->sumSd[k];
        return s;
    }
    default:
        return a->sign * fRuleValue(x, &a->rule);
    }
}

static inline int fAtomPass(FCtx *x, const FAtom *tab, int ai, double t)
{
    const FAtom *a = &tab[ai];
    if (a->kind == FA_SUM) {
        double s = 0;
        for (int k = 0; k < a->nsum; k++) {
            s += (fAtomValue(x, tab, a->sumAtom[k]) - a->sumMu[k]) / a->sumSd[k];
            if (k + 1 < a->nsum) {
                if (s + a->remMax[k] < t - 1e-9) return 0;
                if (s + a->remMin[k] >= t + 1e-9) return 1;
            }
        }
        return s == s && s >= t;
    }
    double v = fAtomValue(x, tab, ai);
    return v == v && v >= t;
}

// crunch() as a filter term: with xSkipN (plain = 0) or with the one-draw skip
// done as a plain step (plain = 1), which agree on every seed.
static inline float fCrunchOct0(Xoroshiro *px, uint64_t mlo, uint64_t mhi, int plain)
{
    uint64_t lo = xNextLong(px), hi = xNextLong(px);
    Xoroshiro xr = { lo ^ mlo, hi ^ mhi };
    if (plain) xNextLong(&xr); else xSkipN(&xr, 1);
    return fabsf(((uint32_t)(xNextLong(&xr) >> 32) & 0xFFFFFF) * 5.960464478E-8f - 0.5f);
}
static int fCrunchRef(uint64_t seed, int plain)
{
    Xoroshiro xr;
    xSetSeed(&xr, seed);
    uint64_t lo = xNextLong(&xr), hi = xNextLong(&xr);
    Xoroshiro hn = { lo ^ 0x81bb4d22e8dc168e, hi ^ 0xf1c8b4bea16303cd };
    float a = fCrunchOct0(&hn, 0x0ef68ec68504005e, 0x48b6bf93a2789640, plain);
    if (a > 0.25f) return 0;
    float b = fCrunchOct0(&hn, 0x0ef68ec68504005e, 0x48b6bf93a2789640, plain);
    if (a + b > 0.30f) return 0;
    Xoroshiro en = { lo ^ 0xd02491e6058f6fd8, hi ^ 0x4792512c94c17a80 };
    float c = fCrunchOct0(&en, 0x082fe255f8be6631, 0x4e96119e22dedc81, plain);
    float d = fCrunchOct0(&en, 0x082fe255f8be6631, 0x4e96119e22dedc81, plain);
    return a + b + c + d <= 0.80f;
}

// ------------------------------------------------------------ expressions

enum { FE_LEAF, FE_AND, FE_OR, FE_CRUNCH };

typedef struct {
    uint8_t type, nkid;
    uint8_t kid[F_MAXKID];
    int16_t atom;           // FE_CRUNCH: 0 = as written, 1 = plain skip
    double t;
} FNode;

typedef struct { int n; FNode nd[F_MAXEN]; } FExpr;     // node 0 is the root

static int fExprPass(FCtx *x, const FAtom *tab, const FExpr *E, int i)
{
    const FNode *n = &E->nd[i];
    switch (n->type) {
    case FE_LEAF:
        return fAtomPass(x, tab, n->atom, n->t);
    case FE_AND:
        for (int k = 0; k < n->nkid; k++) if (!fExprPass(x, tab, E, n->kid[k])) return 0;
        return 1;
    case FE_OR:
        for (int k = 0; k < n->nkid; k++) if (fExprPass(x, tab, E, n->kid[k])) return 1;
        return 0;
    default:
        return fCrunchRef(x->seed, n->atom);
    }
}

static void fExprText(const FExpr *E, const FAtom *tab, int i, char *buf, size_t bn, size_t *at)
{
    const FNode *n = &E->nd[i];
    int w = 0;
    if (*at >= bn) return;
    switch (n->type) {
    case FE_LEAF:
        w = snprintf(buf + *at, bn - *at, "%s>=%.17g", tab[n->atom].name, n->t);
        break;
    case FE_CRUNCH:
        w = snprintf(buf + *at, bn - *at, "%s", n->atom ? "crunch_plain_skip()" : "crunch()");
        break;
    default:
        w = snprintf(buf + *at, bn - *at, "(");
        if (w > 0) *at += (size_t)w;
        for (int k = 0; k < n->nkid; k++) {
            if (k) {
                w = snprintf(buf + *at, bn - *at, "%s", n->type == FE_AND ? " & " : " | ");
                if (w > 0) *at += (size_t)w;
            }
            fExprText(E, tab, n->kid[k], buf, bn, at);
            if (*at >= bn) return;
        }
        w = snprintf(buf + *at, bn - *at, ")");
        break;
    }
    if (w > 0) *at += (size_t)w;
    if (*at >= bn) *at = bn - 1;
}

static void fExprToString(const FExpr *E, const FAtom *tab, char *buf, size_t bn)
{
    size_t at = 0;
    buf[0] = 0;
    fExprText(E, tab, 0, buf, bn, &at);
}

typedef struct {
    char name[256];
    char note[2400];
    uint64_t hash;
    int nafter;
    uint64_t after[F_MAXCHAIN];
    FAtom *atom;
    int natom, cap;
    FExpr E;
} Filter;

static const char *f_pp;
static Filter *f_look;

static int fLookup(const char *nm)
{
    for (int i = 0; i < f_look->natom; i++)
        if (!f_look->atom[i].part && !strcmp(f_look->atom[i].name, nm)) return i;
    return -1;
}

static int fParseNode(FExpr *E)
{
    while (*f_pp == ' ') f_pp++;
    if (E->n >= F_MAXEN) return -1;
    int me = E->n++;
    FNode *n = &E->nd[me];
    memset(n, 0, sizeof(*n));
    if (*f_pp == '(') {
        f_pp++;
        n->type = FE_AND;
        for (;;) {
            if (n->nkid >= F_MAXKID) return -1;
            int c = fParseNode(E);
            if (c < 0) return -1;
            n = &E->nd[me];
            n->kid[n->nkid++] = (uint8_t)c;
            while (*f_pp == ' ') f_pp++;
            if (*f_pp == ')') { f_pp++; break; }
            if (*f_pp == '&') n->type = FE_AND;
            else if (*f_pp == '|') n->type = FE_OR;
            else return -1;
            f_pp++;
        }
        n->atom = -1;
        return me;
    }
    if (!strncmp(f_pp, "crunch()", 8)) { n->type = FE_CRUNCH; n->atom = 0; f_pp += 8; return me; }
    if (!strncmp(f_pp, "crunch_plain_skip()", 19)) { n->type = FE_CRUNCH; n->atom = 1; f_pp += 19; return me; }
    const char *ge = strstr(f_pp, ">=");
    if (!ge || ge - f_pp >= 160) return -1;
    char name[160];
    memcpy(name, f_pp, (size_t)(ge - f_pp));
    name[ge - f_pp] = 0;
    int a = fLookup(name);
    if (a < 0) return -1;
    n->type = FE_LEAF;
    n->atom = (int16_t)a;
    char *end;
    n->t = strtod(ge + 2, &end);
    if (end == ge + 2) return -1;
    f_pp = end;
    return me;
}

static int fExprParse(Filter *F, const char *s)
{
    memset(&F->E, 0, sizeof(F->E));
    f_pp = s;
    f_look = F;
    if (fParseNode(&F->E) != 0) return -1;
    while (*f_pp == ' ') f_pp++;
    return *f_pp ? -1 : 0;
}

// ------------------------------------------------------------ filter files

static int fAddAtom(Filter *F)
{
    if (F->natom == F->cap) {
        int nc = F->cap ? F->cap * 2 : 16;
        FAtom *na = realloc(F->atom, (size_t)nc * sizeof(FAtom));
        if (!na) return -1;
        F->atom = na;
        F->cap = nc;
    }
    memset(&F->atom[F->natom], 0, sizeof(FAtom));
    return F->natom++;
}

static void filterFree(Filter *F)
{
    free(F->atom);
    memset(F, 0, sizeof(*F));
}

static void fSumBounds(FAtom *tab, int ai)
{
    FAtom *A = &tab[ai];
    double zmin[F_MAXSUMK], zmax[F_MAXSUMK];
    for (int k = 0; k < A->nsum; k++) {
        const FAtom *c = &tab[A->sumAtom[k]];
        double lo = c->sign > 0 ? 0.0 : -0.5, hi = c->sign > 0 ? 0.5 : 0.0;
        zmin[k] = (lo - A->sumMu[k]) / A->sumSd[k];
        zmax[k] = (hi - A->sumMu[k]) / A->sumSd[k];
    }
    for (int k = 0; k < A->nsum; k++) {
        A->remMin[k] = 0;
        A->remMax[k] = 0;
        for (int j = k + 1; j < A->nsum; j++) { A->remMin[k] += zmin[j]; A->remMax[k] += zmax[j]; }
    }
}

// The definition lines - what the hash covers - written exactly as the tool
// that makes filter files writes them, so a filter's hash matches its file's.
static size_t filterDefText(const Filter *F, char *buf, size_t bn)
{
    size_t at = 0;
    #define F_PUT(...) do { int w_ = snprintf(buf + at, at < bn ? bn - at : 0, __VA_ARGS__); \
        if (w_ > 0) at += (size_t)w_; } while (0)
    if (bn) buf[0] = 0;
    for (int i = 0; i < F->natom; i++) {
        const FAtom *a = &F->atom[i];
        if (a->part) continue;
        if (a->kind == FA_PHASE) {
            F_PUT("phase %s %.17g\n", a->name, a->sign);
        } else if (a->kind == FA_SUM) {
            F_PUT("sum %s\n", a->name);
            for (int k = 0; k < a->nsum; k++) {
                const FAtom *c = &F->atom[a->sumAtom[k]];
                char nm[F_NAMELEN];
                fOffsetName(c->opi, c->ohalf, c->ooct, c->ooff, nm, sizeof(nm));
                F_PUT("  part %s %.17g %.17g %.17g\n", nm, c->sign, a->sumMu[k], a->sumSd[k]);
            }
        } else {
            F_PUT("rule %s %.17g\n", a->name, a->sign);
            if (at < bn) at += fRuleText(&a->rule, "  ", buf + at, bn - at);
        }
    }
    char ex[4096];
    fExprToString(&F->E, F->atom, ex, sizeof(ex));
    F_PUT("expr %s\n", ex);
    #undef F_PUT
    return at;
}

static uint64_t filterHash(const Filter *F)
{
    char *b = malloc(F_DEFTEXT);
    if (!b) return 0;
    filterDefText(F, b, F_DEFTEXT);
    uint64_t h = fFnv64(b);
    free(b);
    return h;
}

static int fCloseAtom(Filter *F, int ai, const char *where)
{
    FAtom *A = &F->atom[ai];
    if (A->kind == FA_SUM) {
        if (A->nsum < 1) { F_LOG("%s: sum %s has no parts\n", where, A->name); return -1; }
        fSumBounds(F->atom, ai);
        return 0;
    }
    if (A->kind == FA_RULE) return fRuleFinish(&A->rule, where);
    return 0;
}

// Reads every filter in a file into out[0..room). Returns how many, or -1.
static int filterRead(const char *path, Filter *out, int room)
{
    FILE *f = fopen(path, "r");
    if (!f) { F_LOG("cannot open filter file %s: %s\n", path, strerror(errno)); return -1; }
    char *line = malloc(F_LINELEN);
    if (!line) { fclose(f); return -1; }
    char where[1600];
    int n = 0, lineNo = 0, head = 0, bad = 0, cur = -1, haveHash = 0;
    uint64_t fileHash = 0;
    Filter *F = NULL;
    while (!bad && fgets(line, F_LINELEN, f)) {
        lineNo++;
        snprintf(where, sizeof(where), "%s line %d", path, lineNo);
        char *e = line + strlen(line);
        while (e > line && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' ')) *--e = 0;
        char *p = line;
        int indented = (*p == ' ' || *p == '\t');
        while (*p == ' ' || *p == '\t') p++;
        if (!*p || *p == '#') continue;
        if (!head) {
            if (strcmp(p, F_HEAD) != 0) {
                F_LOG("%s is not a filter file (the first line should be \"%s\")\n", path, F_HEAD);
                bad = 1;
            }
            head = 1;
            continue;
        }
        if (indented) {
            if (!F || cur < 0) { F_LOG("%s: an indented line that belongs to no sum or rule\n", where); bad = 1; break; }
            if (F->atom[cur].kind == FA_SUM) {
                char nm[F_NAMELEN];
                double sg, mu, sd;
                int pi, half, oct, off;
                if (sscanf(p, "part %47s %lf %lf %lf", nm, &sg, &mu, &sd) != 4 ||
                    fParseOffsetName(nm, &pi, &half, &oct, &off) != 0 || F->atom[cur].nsum >= F_MAXSUMK || !(sd > 0)) {
                    F_LOG("%s: bad sum part\n", where);
                    bad = 1;
                    break;
                }
                int c = fAddAtom(F);
                if (c < 0) { bad = 1; break; }
                FAtom *C = &F->atom[c], *A = &F->atom[cur];
                snprintf(C->name, sizeof(C->name), "phase.%s", nm);
                C->kind = FA_PHASE;
                C->part = 1;
                C->sign = sg;
                C->opi = pi; C->ohalf = half; C->ooct = oct; C->ooff = off;
                A->sumAtom[A->nsum] = c;
                A->sumMu[A->nsum] = mu;
                A->sumSd[A->nsum] = sd;
                A->nsum++;
            } else if (fRuleLine(&F->atom[cur].rule, p, where) != 0) {
                F_LOG("%s: not a rule line\n", where);
                bad = 1;
                break;
            }
            continue;
        }
        if (F && cur >= 0 && fCloseAtom(F, cur, where) != 0) { bad = 1; break; }
        cur = -1;
        char word[32];
        if (sscanf(p, "%31s", word) != 1) continue;
        const char *rest = p + strlen(word);
        while (*rest == ' ') rest++;
        if (!strcmp(word, "filter")) {
            if (F) { F_LOG("%s: a new filter starts before the last one's \"end\"\n", where); bad = 1; break; }
            if (n >= room) { F_LOG("%s: more than %d filters in all\n", where, F_MAXCHAIN); bad = 1; break; }
            F = &out[n];
            memset(F, 0, sizeof(*F));
            snprintf(F->name, sizeof(F->name), "%s", rest);
            haveHash = 0;
        } else if (!F) {
            F_LOG("%s: \"%s\" outside a filter\n", where, word);
            bad = 1;
        } else if (!strcmp(word, "note")) {
            snprintf(F->note, sizeof(F->note), "%s", rest);
        } else if (!strcmp(word, "after")) {
            if (F->nafter < F_MAXCHAIN) F->after[F->nafter++] = strtoull(rest, NULL, 16);
        } else if (!strcmp(word, "hash")) {
            fileHash = strtoull(rest, NULL, 16);
            haveHash = 1;
        } else if (!strcmp(word, "phase") || !strcmp(word, "sum") || !strcmp(word, "rule")) {
            char nm[160];
            double sg = 1;
            int got = sscanf(rest, "%159s %lf", nm, &sg);
            f_look = F;
            if (got < 1 || fLookup(nm) >= 0) { F_LOG("%s: missing or repeated atom name\n", where); bad = 1; break; }
            int a = fAddAtom(F);
            if (a < 0) { bad = 1; break; }
            FAtom *A = &F->atom[a];
            snprintf(A->name, sizeof(A->name), "%s", nm);
            A->sign = sg;
            if (word[0] == 'p') {
                int pi, half, oct, off;
                if (got != 2 || strncmp(nm, "phase.", 6) || fParseOffsetName(nm + 6, &pi, &half, &oct, &off) != 0) {
                    F_LOG("%s: bad phase line\n", where);
                    bad = 1;
                    break;
                }
                A->kind = FA_PHASE;
                A->opi = pi; A->ohalf = half; A->ooct = oct; A->ooff = off;
            } else if (word[0] == 's') {
                A->kind = FA_SUM;
                A->sign = 1;
                cur = a;
            } else {
                if (got != 2) { F_LOG("%s: a rule line needs a sign\n", where); bad = 1; break; }
                A->kind = FA_RULE;
                fRuleDefaults(&A->rule, -1);
                cur = a;
            }
        } else if (!strcmp(word, "expr")) {
            if (fExprParse(F, rest) != 0) {
                F_LOG("%s: the expression does not parse, or reads an atom not defined above it\n", where);
                bad = 1;
            }
        } else if (!strcmp(word, "end")) {
            if (!F->E.n) { F_LOG("%s: filter \"%s\" has no expr\n", where, F->name); bad = 1; break; }
            F->hash = filterHash(F);
            if (haveHash && fileHash != F->hash)
                F_LOG("note: filter \"%s\" in %s was changed after it was written\n", F->name, path);
            n++;
            F = NULL;
        } else {
            F_LOG("%s: unknown line \"%s\"\n", where, word);
            bad = 1;
        }
    }
    free(line);
    fclose(f);
    if (!bad && F) { F_LOG("%s: the last filter has no \"end\"\n", path); bad = 1; }
    if (!bad && !n) { F_LOG("%s holds no filter\n", path); bad = 1; }
    if (bad) {
        for (int i = 0; i <= n && i < room; i++) filterFree(&out[i]);
        return -1;
    }
    return n;
}

static inline int filterPass(FCtx *x, const Filter *F)
{
    return fExprPass(x, F->atom, &F->E, 0);
}

#endif
