// gpu_host.c - see gpu_host.h.

#include "gpu_host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

GpuApi GPU;

// ------------------------------------------------------------------ loading

#ifdef _WIN32
static HMODULE g_lib;
static void *libSym(const char *name) { return (void *)GetProcAddress(g_lib, name); }

static int libLoad(char *msg, size_t n)
{
    char dir[MAX_PATH + 16];
    DWORD len = GetModuleFileNameA(NULL, dir, MAX_PATH);
    if (!len || len >= MAX_PATH) { snprintf(msg, n, "cannot locate the program"); return -1; }
    char *cut = strrchr(dir, 92);
    if (cut) *cut = 0;
    char path[MAX_PATH + 64];
    snprintf(path, sizeof(path), "%s\\gpu_finder.dll", dir);
    if (GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES) {
        snprintf(msg, n, "gpu_finder.dll not found beside the program (build it with: make gpu)");
        return -1;
    }
    g_lib = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_lib) { snprintf(msg, n, "gpu_finder.dll could not be loaded (error %lu)", GetLastError()); return -1; }
    return 0;
}
#else
static void *g_lib;
static void *libSym(const char *name) { return dlsym(g_lib, name); }

static int libLoad(char *msg, size_t n)
{
    g_lib = dlopen("./gpu_finder.so", RTLD_NOW);
    if (!g_lib) { snprintf(msg, n, "gpu_finder.so could not be loaded: %s", dlerror()); return -1; }
    return 0;
}
#endif

static int g_ready = 0;

static void closeAtExit(void)
{
    if (g_ready && GPU.close) GPU.close();
    g_ready = 0;
}

int gpuOpen(int device, char *msg, size_t msgLen)
{
    if (g_ready) return 0;
    if (!g_lib && libLoad(msg, msgLen) != 0) return -1;
    #define BIND(field, name) GPU.field = (__typeof__(GPU.field))libSym(name); \
        if (!GPU.field) { snprintf(msg, msgLen, "gpu_finder.dll lacks %s - rebuild it with: make gpu", name); return -1; }
    BIND(version, "gf_version");
    if (GPU.version() != GF_VERSION) {
        snprintf(msg, msgLen, "gpu_finder.dll is version %d, the finder expects %d - rebuild it with: make gpu",
                 GPU.version(), GF_VERSION);
        return -1;
    }
    BIND(open, "gf_open");
    BIND(close, "gf_close");
    BIND(error, "gf_error");
    BIND(setup, "gf_setup");
    BIND(gate, "gf_gate");
    BIND(run, "gf_run");
    #undef BIND
    if (GPU.open(device, msg, (int)msgLen) != 0) return -1;
    g_ready = 1;
    atexit(closeAtExit);
    return 0;
}

// --------------------------------------------------------------- layouts

static const uint64_t MD5_OCTAVE[17][2] = {
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

// lib/biomenoise.c init_climate_seed, normal (not large) biomes
typedef struct { int np; uint64_t mlo, mhi; int omin, len; double amp[9]; } ClimateSpec;

static const ClimateSpec SPEC[] = {
    { NP_TEMPERATURE,     0x5c7e6b29735f0d7f, 0xf7d86f1bbc734988, -10, 6, {1.5, 0, 1, 0, 0, 0} },
    { NP_HUMIDITY,        0x81bb4d22e8dc168e, 0xf1c8b4bea16303cd,  -8, 6, {1, 1, 0, 0, 0, 0} },
    { NP_CONTINENTALNESS, 0x83886c9d0ae3a662, 0xafa638a61b42e8ad,  -9, 9, {1, 1, 2, 2, 2, 1, 1, 1, 1} },
    { NP_EROSION,         0xd02491e6058f6fd8, 0x4792512c94c17a80,  -9, 5, {1, 1, 0, 1, 1} },
    { NP_WEIRDNESS,       0xefc8ef4d36102b34, 0x1beeeb324a0f24ea,  -7, 6, {1, 2, 1, 0, 0, 0} },
    { NP_SHIFT,           0x080518cf6af25384, 0x3f3dfb40a54febd5,  -3, 4, {1, 1, 1, 0} },
};

int gpuAddStream(GfLayout *L, int np, int nA, int nB)
{
    const ClimateSpec *cs = NULL;
    for (size_t i = 0; i < sizeof(SPEC) / sizeof(SPEC[0]); i++) if (SPEC[i].np == np) cs = &SPEC[i];
    if (!cs || L->nstream >= GF_MAXSTREAM) return -1;
    int live[9], nlive = 0;
    for (int i = 0; i < cs->len; i++) if (cs->amp[i] != 0) live[nlive++] = i;
    if (nA > nlive) nA = nlive;
    if (nB > nlive) nB = nlive;
    if (nA < 0) nA = 0;
    if (nB < 0) nB = 0;
    if (L->nslot + nA + nB > GF_MAXSLOT) return -1;

    // The amplitudes and lacunarities do not depend on the seed. setBiomeSeed
    // rather than setClimateParaSeed: the latter reads NP_SHIFT as NP_DEPTH.
    BiomeNoise *bn = malloc(sizeof(BiomeNoise));
    if (!bn) return -1;
    setBiomeSeed(bn, 0, 0);
    const DoublePerlinNoise *dp = bn->climate + np;

    int s = L->nstream;
    GfStream *S = &L->stream[s];
    memset(S, 0, sizeof(*S));
    S->mlo = cs->mlo;
    S->mhi = cs->mhi;
    S->amp = dp->amplitude;
    for (int half = 0; half < 2; half++) {
        int n = half ? nB : nA;
        const OctaveNoise *on = half ? &dp->octB : &dp->octA;
        if (half) { S->slotB = L->nslot; S->nB = n; }
        else      { S->slotA = L->nslot; S->nA = n; }
        for (int o = 0; o < n; o++) {
            GfSlot *sl = &L->slot[L->nslot++];
            memset(sl, 0, sizeof(*sl));
            sl->mlo = MD5_OCTAVE[16 + cs->omin + live[o]][0];
            sl->mhi = MD5_OCTAVE[16 + cs->omin + live[o]][1];
            sl->amplitude = on->octaves[o].amplitude;
            sl->lacunarity = on->octaves[o].lacunarity;
            sl->stream = s;
            sl->half = half;
        }
    }
    free(bn);
    L->nstream++;
    return s;
}

int gpuFlattenSpline(const BiomeNoise *bn, GfSpline *out, int max, int *root)
{
    const SplineStack *ss = &bn->ss;
    int n = ss->len + ss->flen;
    if (n > max) return -1;
    memset(out, 0, sizeof(GfSpline) * (size_t)n);
    for (int i = 0; i < ss->len; i++) {
        const Spline *sp = &ss->stack[i];
        GfSpline *g = &out[i];
        g->len = sp->len;
        g->typ = sp->typ;
        if (sp->len == 1) memcpy(&g->val, &sp->typ, sizeof(float));   // read as a FixSpline, as getSpline does
        for (int k = 0; k < sp->len && k < 12; k++) {
            g->loc[k] = sp->loc[k];
            g->der[k] = sp->der[k];
            const char *v = (const char *)sp->val[k];
            const char *s0 = (const char *)ss->stack, *s1 = (const char *)(ss->stack + ss->len);
            const char *f0 = (const char *)ss->fstack, *f1 = (const char *)(ss->fstack + ss->flen);
            if (v >= s0 && v < s1) g->kid[k] = (int)((const Spline *)(void *)v - ss->stack);
            else if (v >= f0 && v < f1) g->kid[k] = ss->len + (int)((const FixSpline *)(void *)v - ss->fstack);
            else return -1;
        }
    }
    for (int i = 0; i < ss->flen; i++) {
        out[ss->len + i].len = 1;
        out[ss->len + i].val = ss->fstack[i].val;
    }
    const char *r = (const char *)bn->sp;
    if (r < (const char *)ss->stack || r >= (const char *)(ss->stack + ss->len)) return -1;
    *root = (int)(bn->sp - ss->stack);
    return n;
}
