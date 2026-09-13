// gpu_finder.cu - the finder's gates as CUDA kernels. See gpu_finder.h.
//
// The GPU is given only seeds. Each thread seeds xoroshiro and draws what its
// gate reads - lattice offsets, or the perlin octaves it samples - and works
// from there. After each gate the passing seeds are compacted into a new list.
//
// Device code cannot recurse, so the depth spline and the biome tree search
// (lib/biomenoise.c) are written as loops.
//
// Kernel launches are sized to take 0.3 - 1.2 s: Windows resets a display GPU
// whose single piece of work runs for about two seconds.

#define GF_BUILDING
#include "gpu_finder.h"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

#define BLOCK 256
#define DP_F  (337.0 / 331.0)
#define CHUNK 4096

enum { S_TEMP, S_HUM, S_CONT, S_EROS, S_WEIRD, S_SHIFT };
enum { ST_GATE, ST_TE, ST_PRE, ST_COARSE, ST_GRID, ST_FULL, NSTAGE };

// ------------------------------------------------------------------ errors

static char g_err[1024];

static void setErr(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}

#define TRY(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    setErr("CUDA %s (%d) at gpu_finder.cu:%d", cudaGetErrorString(e_), (int)e_, __LINE__); return -1; } } while (0)
#define LAUNCHED() TRY(cudaGetLastError())

static uint32_t blocksFor(int64_t n)
{
    int64_t b = (n + BLOCK - 1) / BLOCK;
    return (uint32_t)(b < 1 ? 1 : b);
}

static double nowMs(void)
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// A device allocation that frees itself.
struct DevBuf {
    void *p = nullptr;
    size_t n = 0;
    int alloc(size_t bytes) {
        free();
        if (!bytes) bytes = 1;
        cudaError_t e = cudaMalloc(&p, bytes);
        if (e != cudaSuccess) {
            setErr("could not allocate %.1f MB on the GPU: %s", bytes / 1048576.0, cudaGetErrorString(e));
            p = nullptr;
            return -1;
        }
        n = bytes;
        return 0;
    }
    int ensure(size_t bytes) { return n >= bytes && p ? 0 : alloc(bytes); }
    int upload(const void *src, size_t bytes) {
        if (alloc(bytes) != 0) return -1;
        TRY(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
        return 0;
    }
    void free() { if (p) cudaFree(p); p = nullptr; n = 0; }
    ~DevBuf() { free(); }
    template <typename T> T *as() const { return (T *)p; }
};

// ------------------------------------------------------------- xoroshiro

struct Xr { uint64_t lo, hi; };

__host__ __device__ static inline uint64_t rotl64(uint64_t x, int b)
{
    return (x << b) | (x >> (64 - b));
}
__host__ __device__ static inline void xSetSeed(Xr &xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr.lo = l;
    xr.hi = h;
}
__host__ __device__ static inline uint64_t xNextLong(Xr &xr)
{
    uint64_t l = xr.lo, h = xr.hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr.lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr.hi = rotl64(h, 28);
    return n;
}
__host__ __device__ static inline int xNextInt(Xr &xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n) {
        while ((uint32_t)r < (~n + 1) % n)
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    }
    return (int)(r >> 32);
}
__host__ __device__ static inline double xNextDouble(Xr &xr)
{
    return (xNextLong(xr) >> (64 - 53)) * 1.1102230246251565E-16;
}

// --------------------------------------------------------------- octaves

// One perlin octave of one seed: lib/noise.h's PerlinNoise without the
// amplitude and lacunarity, which do not depend on the seed and live in the
// layout.
struct DevOct {
    double a, b, c;
    double d2, t2;
    uint8_t h2;
    uint8_t d[257];
};

// xPerlinInit
__device__ static void initOctave(DevOct &o, Xr q)
{
    o.a = xNextDouble(q) * 256.0;
    o.b = xNextDouble(q) * 256.0;
    o.c = xNextDouble(q) * 256.0;
    for (int i = 0; i < 256; i++) o.d[i] = (uint8_t)i;
    for (int i = 0; i < 256; i++) {
        int j = xNextInt(q, 256 - i) + i;
        uint8_t n = o.d[i];
        o.d[i] = o.d[j];
        o.d[j] = n;
    }
    o.d[256] = o.d[0];
    double i2 = floor(o.b);
    double d2 = o.b - i2;
    o.h2 = (uint8_t)(int)i2;
    o.d2 = d2;
    o.t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
}

// setBiomeSeed / setClimateParaSeed for the slots of a layout. Each octave is
// seeded from its half's two longs and its own md5, so any subset of octaves
// initialises exactly.
__device__ static void initSeed(const GfLayout *L, uint64_t seed, DevOct *out)
{
    Xr r;
    xSetSeed(r, seed);
    uint64_t xlo = xNextLong(r), xhi = xNextLong(r);
    uint64_t hl[GF_MAXSTREAM][2], hh[GF_MAXSTREAM][2];
    for (int s = 0; s < L->nstream; s++) {
        Xr p = { xlo ^ L->stream[s].mlo, xhi ^ L->stream[s].mhi };
        hl[s][0] = xNextLong(p);
        hh[s][0] = xNextLong(p);
        hl[s][1] = xNextLong(p);
        hh[s][1] = xNextLong(p);
    }
    for (int k = 0; k < L->nslot; k++) {
        const GfSlot &sl = L->slot[k];
        Xr q = { hl[sl.stream][sl.half] ^ sl.mlo, hh[sl.stream][sl.half] ^ sl.mhi };
        initOctave(out[k], q);
    }
}

struct InitArgs {
    const GfLayout *L;
    const uint64_t *seeds;
    int64_t n;
    DevOct *out;                // [n][nslot]
};

__global__ void kInit(InitArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    initSeed(A.L, A.seeds[i], A.out + i * A.L->nslot);
}

// --------------------------------------------------------------- sampling

__device__ static inline double indexedLerp(uint8_t idx, double a, double b, double c)
{
    switch (idx & 0xf) {
    case 0:  return  a + b;
    case 1:  return -a + b;
    case 2:  return  a - b;
    case 3:  return -a - b;
    case 4:  return  a + c;
    case 5:  return -a + c;
    case 6:  return  a - c;
    case 7:  return -a - c;
    case 8:  return  b + c;
    case 9:  return -b + c;
    case 10: return  b - c;
    case 11: return -b - c;
    case 12: return  a + b;
    case 13: return -b + c;
    case 14: return -a + b;
    default: return -b - c;
    }
}

__device__ static inline double lerpD(double part, double from, double to)
{
    return from + part * (to - from);
}

// samplePerlin with yamp = 0, from the point where the y lattice is known.
__device__ static double perlinCore(const DevOct &o, double d1, double d2, uint8_t h2, double t2, double d3)
{
    d1 += o.a;
    d3 += o.c;
    double i1 = floor(d1);
    double i3 = floor(d3);
    d1 -= i1;
    d3 -= i3;
    uint8_t h1 = (uint8_t)(int)i1;
    uint8_t h3 = (uint8_t)(int)i3;
    double t1 = d1 * d1 * d1 * (d1 * (d1 * 6.0 - 15.0) + 10.0);
    double t3 = d3 * d3 * d3 * (d3 * (d3 * 6.0 - 15.0) + 10.0);

    const uint8_t *idx = o.d;
    uint8_t v1a = idx[h1], v1b = idx[h1 + 1];
    v1a += h2;
    v1b += h2;
    uint8_t v2a = idx[v1a], v2b = idx[v1a + 1];
    uint8_t v3a = idx[v1b], v3b = idx[v1b + 1];
    v2a += h3; v2b += h3;
    v3a += h3; v3b += h3;
    uint8_t v4a = idx[v2a], v4b = idx[v2a + 1];
    uint8_t v5a = idx[v2b], v5b = idx[v2b + 1];
    uint8_t v6a = idx[v3a], v6b = idx[v3a + 1];
    uint8_t v7a = idx[v3b], v7b = idx[v3b + 1];

    double l1 = indexedLerp(v4a, d1,     d2,     d3);
    double l5 = indexedLerp(v4b, d1,     d2,     d3 - 1);
    double l2 = indexedLerp(v6a, d1 - 1, d2,     d3);
    double l6 = indexedLerp(v6b, d1 - 1, d2,     d3 - 1);
    double l3 = indexedLerp(v5a, d1,     d2 - 1, d3);
    double l7 = indexedLerp(v5b, d1,     d2 - 1, d3 - 1);
    double l4 = indexedLerp(v7a, d1 - 1, d2 - 1, d3);
    double l8 = indexedLerp(v7b, d1 - 1, d2 - 1, d3 - 1);

    l1 = lerpD(t1, l1, l2);
    l3 = lerpD(t1, l3, l4);
    l5 = lerpD(t1, l5, l6);
    l7 = lerpD(t1, l7, l8);
    l1 = lerpD(t2, l1, l3);
    l5 = lerpD(t2, l5, l7);
    return lerpD(t3, l1, l5);
}

// lib/noise.c samplePerlin(noise, d1, d2, d3, 0, 0)
__device__ static inline double samplePerlin(const DevOct &o, double d1, double d2, double d3)
{
    if (d2 == 0.0)
        return perlinCore(o, d1, o.d2, o.h2, o.t2, d3);
    d2 += o.b;
    double i2 = floor(d2);
    d2 -= i2;
    uint8_t h2 = (uint8_t)(int)i2;
    double t2 = d2 * d2 * d2 * (d2 * (d2 * 6.0 - 15.0) + 10.0);
    return perlinCore(o, d1, d2, h2, t2, d3);
}

// sampleOctave over a layout's slots
__device__ static double octaveSum(const GfLayout *L, const DevOct *o, int first, int n, double x, double y, double z)
{
    double v = 0;
    for (int i = 0; i < n; i++) {
        const GfSlot &sl = L->slot[first + i];
        double lf = sl.lacunarity;
        double pv = samplePerlin(o[first + i], x * lf, y * lf, z * lf);
        v += sl.amplitude * pv;
    }
    return v;
}

// sampleDoublePerlin
__device__ static double doublePerlin(const GfLayout *L, const DevOct *o, int s, double x, double y, double z)
{
    const GfStream &S = L->stream[s];
    double v = 0;
    v += octaveSum(L, o, S.slotA, S.nA, x, y, z);
    v += octaveSum(L, o, S.slotB, S.nB, x * DP_F, y * DP_F, z * DP_F);
    return v * S.amp;
}

// ------------------------------------------------------------- biome tables

struct DevTables {
    int32_t order, len, nsteps, nparam;
    uint32_t steps[16];
    const int32_t *param;           // device memory
    const uint64_t *nodes;          // device memory
    const GfSpline *spline;         // device memory
    int32_t nspline, splineRoot;
    int16_t idToIndex[256];
    int32_t numBiomes;
    int32_t sampleRange, teStride, preStride, coarseScale, sampleY, sampleStep, trueStep;
    double teBand[4], teWeight[5];
};

__device__ static inline uint64_t npDist(const DevTables *T, const uint64_t *np, int idx)
{
    uint64_t ds = 0, node = T->nodes[idx];
    for (uint32_t i = 0; i < 6; i++) {
        int k = (int)((node >> 8 * i) & 0xFF);
        uint64_t a = np[i] - (uint64_t)(int64_t)T->param[2 * k + 1];
        uint64_t b = (uint64_t)(int64_t)T->param[2 * k + 0] - np[i];
        uint64_t d = (int64_t)a > 0 ? a : (int64_t)b > 0 ? b : 0;
        d = d * d;
        ds += d;
    }
    return ds;
}

// lib/biomenoise.c get_resulting_node, with the recursion on an explicit stack.
__device__ static int resultingNode(const DevTables *T, const uint64_t *np, int idx, int alt, uint64_t ds)
{
    struct Frame { uint64_t ds, dsInner; int leaf, depth, i; uint32_t step; uint16_t inner; };
    Frame st[16];
    int top = -1, ret = 0, mode = 0;        // 0: enter a call  1: run the top frame  2: return ret to it
    int eIdx = idx, eAlt = alt, eDepth = 0;
    uint64_t eDs = ds;
    for (;;) {
        if (mode == 0) {
            if (T->steps[eDepth] == 0) {
                ret = eIdx;
                mode = 2;
                continue;
            }
            uint32_t step;
            int depth = eDepth;
            do {
                step = T->steps[depth];
                depth++;
            } while ((uint32_t)eIdx + step >= (uint32_t)T->len);
            top++;
            st[top].ds = eDs;
            st[top].leaf = eAlt;
            st[top].depth = depth;
            st[top].step = step;
            st[top].inner = (uint16_t)(T->nodes[eIdx] >> 48);
            st[top].i = 0;
            mode = 1;
            continue;
        }
        if (mode == 2) {
            if (top < 0) return ret;
            Frame &F = st[top];
            uint64_t dsLeaf2 = ((int)F.inner == ret) ? F.dsInner : npDist(T, np, ret);
            if (dsLeaf2 < F.ds) {
                F.ds = dsLeaf2;
                F.leaf = ret;
            }
            F.inner = (uint16_t)(F.inner + F.step);
            F.i++;
            if (F.inner >= T->len) F.i = T->order;
            mode = 1;
            continue;
        }
        Frame &F = st[top];
        if (F.i >= T->order) {
            ret = F.leaf;
            top--;
            mode = 2;
            continue;
        }
        uint64_t dsInner = npDist(T, np, F.inner);
        if (dsInner < F.ds) {
            F.dsInner = dsInner;
            eIdx = F.inner;
            eAlt = F.leaf;
            eDs = F.ds;
            eDepth = F.depth;
            mode = 0;
            continue;
        }
        F.inner = (uint16_t)(F.inner + F.step);
        F.i++;
        if (F.inner >= T->len) F.i = T->order;
    }
}

__device__ static int climateToBiome(const DevTables *T, const uint64_t *np, uint64_t *dat)
{
    int idx;
    if (dat) {
        int alt = (int)*dat;
        uint64_t ds = npDist(T, np, alt);
        idx = resultingNode(T, np, 0, alt, ds);
        *dat = (uint64_t)idx;
    } else {
        idx = resultingNode(T, np, 0, 0, (uint64_t)-1);
    }
    return (int)((T->nodes[idx] >> 48) & 0xFF);
}

// lib/biomenoise.c getSpline, with the recursion on an explicit stack.
__device__ static float getSpline(const DevTables *T, const float *vals)
{
    struct Frame { int sp, stage, i; float n; };
    Frame st[16];
    int top = 0, mode = 0;                  // 0: run the top frame  1: hand ret to it
    float ret = 0;
    st[0].sp = T->splineRoot;
    st[0].stage = 0;
    for (;;) {
        Frame &F = st[top];
        const GfSpline &S = T->spline[F.sp];
        if (mode == 1) {
            float f = vals[S.typ];
            if (F.stage == 10) {
                ret = ret + S.der[F.i] * (f - S.loc[F.i]);
            } else if (F.stage == 1) {
                F.n = ret;
                F.stage = 2;
                top++;
                st[top].sp = S.kid[F.i];
                st[top].stage = 0;
                mode = 0;
                continue;
            } else {
                int i = F.i;
                float g = S.loc[i - 1];
                float h = S.loc[i];
                float k = (f - g) / (h - g);
                float l = S.der[i - 1];
                float m = S.der[i];
                float n = F.n;
                float o = ret;
                float p = l * (h - g) - (o - n);
                float q = -m * (h - g) + (o - n);
                float r = lerpD(k, n, o) + k * (1.0F - k) * lerpD(k, p, q);
                ret = r;
            }
            if (top == 0) return ret;
            top--;
            continue;
        }
        if (S.len == 1) {
            ret = S.val;
            if (top == 0) return ret;
            top--;
            mode = 1;
            continue;
        }
        float f = vals[S.typ];
        int i;
        for (i = 0; i < S.len; i++)
            if (S.loc[i] >= f)
                break;
        if (i == 0 || i == S.len) {
            if (i) i--;
            F.i = i;
            F.stage = 10;
            top++;
            st[top].sp = S.kid[i];
            st[top].stage = 0;
            continue;
        }
        F.i = i;
        F.stage = 1;
        top++;
        st[top].sp = S.kid[i - 1];
        st[top].stage = 0;
    }
}

__device__ static inline float peaksAndValleys(float weirdness)
{
    return -(fabs(fabs(weirdness) - 0.6666667F) - 0.33333334F) * 3.0F;
}

// lib/biomenoise.c sampleBiomeNoise for a layout holding the five climate
// noises in stream order (and the shift noise as stream 5 when shifting).
__device__ static int sampleBiome(const GfLayout *L, const DevOct *o, const DevTables *T,
                                  int x, int y, int z, uint64_t *dat, int noShift)
{
    float t = 0, h = 0, c = 0, e = 0, d = 0, w = 0;
    double px = x, pz = z;
    if (!noShift) {
        px += doublePerlin(L, o, S_SHIFT, x, 0, z) * 4.0;
        pz += doublePerlin(L, o, S_SHIFT, z, x, 0) * 4.0;
    }
    c = doublePerlin(L, o, S_CONT, px, 0, pz);
    e = doublePerlin(L, o, S_EROS, px, 0, pz);
    w = doublePerlin(L, o, S_WEIRD, px, 0, pz);
    float np_param[] = { c, e, peaksAndValleys(w), w };
    double off = getSpline(T, np_param) + 0.015F;
    d = 1.0 - (y * 4) / 128.0 - 83.0 / 160.0 + off;
    t = doublePerlin(L, o, S_TEMP, px, 0, pz);
    h = doublePerlin(L, o, S_HUM, px, 0, pz);
    uint64_t np[6];
    np[0] = (uint64_t)(int64_t)(10000.0F * t);
    np[1] = (uint64_t)(int64_t)(10000.0F * h);
    np[2] = (uint64_t)(int64_t)(10000.0F * c);
    np[3] = (uint64_t)(int64_t)(10000.0F * e);
    np[4] = (uint64_t)(int64_t)(10000.0F * d);
    np[5] = (uint64_t)(int64_t)(10000.0F * w);
    return climateToBiome(T, np, dat);
}

// main.c arbFromCounts
__device__ static double arbFromCounts(const int32_t *counts, int nb)
{
    double total = 0.0;
    for (int i = 0; i < nb; i++) total += (double)counts[i];
    if (total <= 0.0) return 0.0;
    double H = 0.0;
    for (int i = 0; i < nb; i++) {
        if (counts[i]) {
            double p = (double)counts[i] / total;
            H += p * log(p);
        }
    }
    H = -H;
    double normH = H / log((double)nb);
    return normH * normH * 100.0;
}

__device__ static inline void countBiome(const DevTables *T, int32_t *counts, int b)
{
    if (b >= 0 && b < 256) {
        int idx = T->idToIndex[b];
        if (idx >= 0) counts[idx]++;
    }
}

__device__ static inline int64_t llroundD(double x)
{
    double r = trunc(x);
    if (fabs(x - r) >= 0.5) r += (x < 0 ? -1.0 : 1.0);
    return (int64_t)r;
}

// ------------------------------------------------------------ stage values

// main.c temp_even: the te layout's stream 0 is temperature
__device__ static double teValue(const GfLayout *L, const DevOct *o, const DevTables *T)
{
    int cnt[5] = { 0, 0, 0, 0, 0 }, tot = 0, half = T->sampleRange / 2;
    for (int z = -half; z <= half; z += T->teStride)
        for (int x = -half; x <= half; x += T->teStride) {
            double v = doublePerlin(L, o, 0, x / 4.0, 0, z / 4.0);
            int b = 0;
            while (b < 4 && v >= T->teBand[b]) b++;
            cnt[b]++;
            tot++;
        }
    double H = 0.0, Wt = 0.0;
    for (int b = 0; b < 5; b++) Wt += T->teWeight[b];
    for (int b = 0; b < 5; b++)
        if (cnt[b]) {
            double p = (double)cnt[b] / tot;
            H += -p * log(p / T->teWeight[b]);
        }
    return H / log(Wt);
}

// main.c prefilter
__device__ static double preValue(const GfLayout *L, const DevOct *o, const DevTables *T)
{
    int32_t counts[GF_MAXBIOME];
    for (int i = 0; i < GF_MAXBIOME; i++) counts[i] = 0;
    int half = T->sampleRange / 2;
    for (int bz = -half; bz <= half; bz += T->preStride)
        for (int bx = -half; bx <= half; bx += T->preStride) {
            double qx = bx / 4.0, qz = bz / 4.0;
            uint64_t np[6];
            np[0] = (uint64_t)llroundD(doublePerlin(L, o, S_TEMP, qx, 0, qz) * 10000.0);
            np[1] = (uint64_t)llroundD(doublePerlin(L, o, S_HUM, qx, 0, qz) * 10000.0);
            np[2] = (uint64_t)llroundD(doublePerlin(L, o, S_CONT, qx, 0, qz) * 10000.0);
            np[3] = (uint64_t)llroundD(doublePerlin(L, o, S_EROS, qx, 0, qz) * 10000.0);
            np[4] = 0;
            np[5] = (uint64_t)llroundD(doublePerlin(L, o, S_WEIRD, qx, 0, qz) * 10000.0);
            uint64_t dat = 0;
            countBiome(T, counts, climateToBiome(T, np, &dat));
        }
    return arbFromCounts(counts, T->numBiomes);
}

// main.c coarse: genBiomes at COARSE_SCALE, which samples without the shift
// noise and carries the last leaf from cell to cell
__device__ static double coarseValue(const GfLayout *L, const DevOct *o, const DevTables *T)
{
    int32_t counts[GF_MAXBIOME];
    for (int i = 0; i < GF_MAXBIOME; i++) counts[i] = 0;
    int W = T->sampleRange / T->coarseScale + 1;
    int xb = -(T->sampleRange / 2) / T->coarseScale, zb = xb, yb = T->sampleY / 4;
    int scale = T->coarseScale > 4 ? T->coarseScale / 4 : 1;
    int mid = scale / 2;
    uint64_t dat = 0;
    for (int j = 0; j < W; j++) {
        int zj = (zb + j) * scale + mid;
        for (int i = 0; i < W; i++) {
            int xi = (xb + i) * scale + mid;
            countBiome(T, counts, sampleBiome(L, o, T, xi, yb, zj, &dat, 1));
        }
    }
    return arbFromCounts(counts, T->numBiomes);
}

// ------------------------------------------------------------ stage kernels

// temp_even, prefilter and coarse straight from the seed, with the octaves a
// seed reads held by the thread itself: far faster than filling a shared
// buffer with them first. A DevOct is about 300 bytes, so temp_even and
// prefilter's usual few octaves get a smaller array.
#define SMALL_SLOTS 10

struct StageArgs {
    int32_t stage;
    const GfLayout *L;
    const DevTables *T;
    const uint64_t *seeds;
    int64_t n;
    double cut;
    uint8_t *flag;              // value >= cut
};

__device__ static inline double tePreValue(const StageArgs &A, const DevOct *o)
{
    return A.stage == ST_TE ? teValue(A.L, o, A.T) : preValue(A.L, o, A.T);
}

__global__ void kTePre(StageArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    DevOct o[SMALL_SLOTS];
    initSeed(A.L, A.seeds[i], o);
    A.flag[i] = tePreValue(A, o) >= A.cut;
}

__global__ void kTePreLarge(StageArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    DevOct o[GF_LOCALSLOT];
    initSeed(A.L, A.seeds[i], o);
    A.flag[i] = tePreValue(A, o) >= A.cut;
}

__global__ void kCoarse(StageArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    DevOct o[GF_LOCALSLOT];
    initSeed(A.L, A.seeds[i], o);
    A.flag[i] = coarseValue(A.L, o, A.T) >= A.cut;
}

// gridscan and fullscan: one thread per (seed, cell), writing the cell's
// scored biome. Cell (i, j) is at x0 + i*dx, z0 + j*dx. Each launch covers
// cells c0 .. c0+ncell of every seed. (One thread walking a seed's cells in
// turn is far slower on a GPU: its cores are many, not fast.)
struct CellArgs {
    const GfLayout *L;
    const DevOct *oct;
    const DevTables *T;
    int64_t nseed, c0, ncell;
    int32_t W, x0, z0, dx, y;
    int8_t *out;                // [nseed][W*W]
};

__global__ void kCells(CellArgs A)
{
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= A.nseed * A.ncell) return;
    int64_t s = t / A.ncell, c = A.c0 + t % A.ncell;
    int x = A.x0 + (int)(c % A.W) * A.dx, z = A.z0 + (int)(c / A.W) * A.dx;
    int b = sampleBiome(A.L, A.oct + s * A.L->nslot, A.T, x, A.y, z, nullptr, 0);
    A.out[s * (int64_t)A.W * A.W + c] = (int8_t)((b >= 0 && b < 256) ? A.T->idToIndex[b] : -1);
}

struct CellReduceArgs {
    const DevTables *T;
    const int8_t *cells;
    int64_t nseed, ncell;
    double *arb;
    int32_t *present;
};

__global__ void kCellReduce(CellReduceArgs A)
{
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= A.nseed) return;
    int32_t counts[GF_MAXBIOME];
    for (int i = 0; i < GF_MAXBIOME; i++) counts[i] = 0;
    const int8_t *c = A.cells + s * A.ncell;
    for (int64_t k = 0; k < A.ncell; k++)
        if (c[k] >= 0) counts[c[k]]++;
    int p = 0;
    for (int i = 0; i < A.T->numBiomes; i++) p += counts[i] != 0;
    A.arb[s] = arbFromCounts(counts, A.T->numBiomes);
    A.present[s] = p;
}

// ------------------------------------------------------------------ gate

// The lattice offsets a seed's filters read, drawn only as far as needed:
// the world seed's two longs, a noise's two longs per half, then a, b, c.
struct Lattice {
    uint64_t seed;
    bool haveBase;
    uint64_t xlo, xhi;
    uint8_t nStream[5];
    uint64_t slo[5][2], shi[5][2];
    uint8_t nd[5][2][2];
    Xr sub[5][2][2];
    double raw[5][2][2][3];
};

__device__ static void latStart(Lattice &x, uint64_t seed)
{
    x.seed = seed;
    x.haveBase = false;
    for (int p = 0; p < 5; p++) {
        x.nStream[p] = 0;
        for (int h = 0; h < 2; h++) { x.nd[p][h][0] = 0; x.nd[p][h][1] = 0; }
    }
}

__device__ static double latOffset(Lattice &x, const GfGate *G, int pi, int half, int oct, int off)
{
    uint8_t &nd = x.nd[pi][half][oct];
    if (nd > off) return x.raw[pi][half][oct][off];
    if (!x.haveBase) {
        Xr r;
        xSetSeed(r, x.seed);
        x.xlo = xNextLong(r);
        x.xhi = xNextLong(r);
        x.haveBase = true;
    }
    if (x.nStream[pi] <= half) {
        Xr r = { x.xlo ^ G->pmd5[pi][0], x.xhi ^ G->pmd5[pi][1] };
        for (int h = 0; h <= half; h++) {
            x.slo[pi][h] = xNextLong(r);
            x.shi[pi][h] = xNextLong(r);
        }
        x.nStream[pi] = (uint8_t)(half + 1);
    }
    Xr &q = x.sub[pi][half][oct];
    if (nd == 0) {
        q.lo = x.slo[pi][half] ^ G->omd5[pi][oct][0];
        q.hi = x.shi[pi][half] ^ G->omd5[pi][oct][1];
    }
    while (nd <= off) {
        x.raw[pi][half][oct][nd] = xNextDouble(q) * 256.0;
        nd++;
    }
    return x.raw[pi][half][oct][off];
}

__device__ static inline double phaseOf(double raw) { return fabs((raw - floor(raw)) - 0.5); }

// main.c crunch() with its one-draw skip as a plain step (the same answer)
__device__ static inline float crunchOct0(Xr &px, uint64_t mlo, uint64_t mhi)
{
    uint64_t lo = xNextLong(px), hi = xNextLong(px);
    Xr xr = { lo ^ mlo, hi ^ mhi };
    xNextLong(xr);
    return fabs(((uint32_t)(xNextLong(xr) >> 32) & 0xFFFFFF) * 5.960464478E-8f - 0.5f);
}
__device__ static inline bool crunchSeed(uint64_t seed)
{
    Xr xr;
    xSetSeed(xr, seed);
    uint64_t lo = xNextLong(xr), hi = xNextLong(xr);
    Xr hn = { lo ^ 0x81bb4d22e8dc168eULL, hi ^ 0xf1c8b4bea16303cdULL };
    float a = crunchOct0(hn, 0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL);
    if (a > 0.25f) return false;
    float b = crunchOct0(hn, 0x0ef68ec68504005eULL, 0x48b6bf93a2789640ULL);
    if (a + b > 0.30f) return false;
    Xr en = { lo ^ 0xd02491e6058f6fd8ULL, hi ^ 0x4792512c94c17a80ULL };
    float c = crunchOct0(en, 0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL);
    float d = crunchOct0(en, 0x082fe255f8be6631ULL, 0x4e96119e22dedc81ULL);
    return a + b + c + d <= 0.80f;
}

// filter.h's fMetricOne and fOrbitM over the rule layout's octaves
__device__ static double octValue(const GfLayout *L, const DevOct *o, int slot, double x, double z)
{
    const GfSlot &sl = L->slot[slot];
    double lf = sl.lacunarity;
    return sl.amplitude * samplePerlin(o[slot], x * lf, 0.0, z * lf);
}

__device__ static double metricOne(const GfLayout *L, const DevOct *o, int s, double bx, double bz, int metric)
{
    const GfStream &S = L->stream[s];
    double qx = bx * 0.25, qz = bz * 0.25;
    double xb = qx * DP_F, zb = qz * DP_F;
    switch (metric) {
    case 0:
        return octValue(L, o, S.slotA, qx, qz) * S.amp;
    case 1:
        return octValue(L, o, S.slotB, xb, zb) * S.amp;
    case 2:
        return (octValue(L, o, S.slotA, qx, qz) + octValue(L, o, S.slotB, xb, zb)) * S.amp;
    case 3: {
        double a = octValue(L, o, S.slotA, qx, qz);
        a += octValue(L, o, S.slotA + 1, qx, qz);
        double b = octValue(L, o, S.slotB, xb, zb);
        b += octValue(L, o, S.slotB + 1, xb, zb);
        return (a + b) * S.amp;
    }
    default: {
        double a = 0, b = 0;
        for (int i = 0; i < S.nA; i++) a += octValue(L, o, S.slotA + i, qx, qz);
        for (int i = 0; i < S.nB; i++) b += octValue(L, o, S.slotB + i, xb, zb);
        return (a + b) * S.amp;
    }
    }
}

__device__ static double orbitValue(const GfLayout *L, const DevOct *o, int s, int sym, int agg, int metric, int bx, int bz)
{
    int ox[8], oz[8], no;
    int x = bx, z = bz;
    if (sym == 0) { ox[0] = x; oz[0] = z; no = 1; }
    else {
        if (x < 0) x = -x;
        if (z < 0) z = -z;
        ox[0] =  x; oz[0] =  z;  ox[1] =  x; oz[1] = -z;
        ox[2] = -x; oz[2] = -z;  ox[3] = -x; oz[3] =  z;
        no = 4;
        if (sym == 2) {
            ox[4] =  z; oz[4] =  x;  ox[5] =  z; oz[5] = -x;
            ox[6] = -z; oz[6] = -x;  ox[7] = -z; oz[7] =  x;
            no = 8;
        }
    }
    double e = agg == 1 ? -INFINITY : agg == 2 ? INFINITY : 0.0;
    for (int k = 0; k < no; k++) {
        double v = metricOne(L, o, s, ox[k], oz[k], metric);
        if (agg == 0) { double a = fabs(v); if (a > e) e = a; }
        else if (agg == 1) { if (v > e) e = v; }
        else { if (v < e) e = v; }
    }
    return e;
}

__device__ static double ruleValue(const GfGate *G, const GfRule *R, const DevOct *o, Lattice &lat)
{
    const GfLayout *L = &G->ruleLayout;
    double s = 0;
    for (int i = 0; i < R->nterm; i++) {
        double v;
        if (R->kind[i] == GF_TK_ORIGIN || R->kind[i] == GF_TK_PHASE) {
            double raw = latOffset(lat, G, R->pi[i], R->bx[i], R->bz[i], R->metric[i]);
            v = R->kind[i] == GF_TK_ORIGIN ? fabs(raw / 256.0 - 0.5) : phaseOf(raw);
        } else if (R->kind[i] == GF_TK_FARNEAR) {
            double nearSum = 0, farSum = 0;
            for (int k = 0; k < 8; k++) {
                double aa = orbitValue(L, o, R->stream[i], R->sym, R->agg, R->metric[i], G->farnear[k], G->farnear[8 + k]);
                double bb = orbitValue(L, o, R->stream[i], R->sym, R->agg, R->metric[i], G->farnear[16 + k], G->farnear[24 + k]);
                nearSum += aa;
                farSum += bb;
            }
            v = farSum / 8 - nearSum / 8;
        } else {
            v = orbitValue(L, o, R->stream[i], R->sym, R->agg, R->metric[i], R->bx[i], R->bz[i]);
        }
        s += R->sign[i] * (v - R->mu[i]) / R->sd[i];
    }
    return s;
}

// A leaf: 0 fails, 1 passes, 2 unknown (a rule, when no octaves are given).
__device__ static int leafValue(const GfGate *G, const GfNode &N, Lattice &lat, const DevOct *o)
{
    if (N.type == GF_N_CRUNCH) return crunchSeed(lat.seed) ? 1 : 0;
    const GfAtom &A = G->atom[N.atom];
    if (A.kind == GF_A_PHASE) {
        double v = A.sign[0] * phaseOf(latOffset(lat, G, A.pi[0], A.half[0], A.oct[0], A.off[0]));
        return (v == v && v >= N.t) ? 1 : 0;
    }
    if (A.kind == GF_A_SUM) {
        double s = 0;
        for (int k = 0; k < A.nsum; k++) {
            double v = A.sign[k] * phaseOf(latOffset(lat, G, A.pi[k], A.half[k], A.oct[k], A.off[k]));
            s += (v - A.mu[k]) / A.sd[k];
        }
        return (s == s && s >= N.t) ? 1 : 0;
    }
    if (!o) return 2;
    double v = A.rsign * ruleValue(G, &G->rule[A.rule], o, lat);
    return (v == v && v >= N.t) ? 1 : 0;
}

// One filter's tree in three-valued logic, stopping early as the CPU does.
__device__ static int treeValue(const GfGate *G, int root, Lattice &lat, const DevOct *o)
{
    struct Frame { int node, k, acc; };
    Frame st[40];
    int top = 0, ret = 0, mode = 0;         // 0: run the top frame  1: hand ret to it
    st[0].node = root;
    st[0].k = -1;
    for (;;) {
        Frame &F = st[top];
        const GfNode &N = G->node[F.node];
        if (mode == 1) {
            if (N.type == GF_N_AND) {
                if (ret == 0) { F.acc = 0; F.k = N.nkid; }
                else if (ret == 2) { F.acc = 2; F.k++; }
                else F.k++;
            } else {
                if (ret == 1) { F.acc = 1; F.k = N.nkid; }
                else if (ret == 2) { F.acc = 2; F.k++; }
                else F.k++;
            }
            mode = 0;
        }
        if (N.type == GF_N_LEAF || N.type == GF_N_CRUNCH) {
            ret = leafValue(G, N, lat, o);
        } else {
            if (F.k < 0) {
                F.acc = N.type == GF_N_AND ? 1 : 0;
                F.k = 0;
            }
            if (F.k < N.nkid) {
                top++;
                st[top].node = N.kid[F.k];
                st[top].k = -1;
                continue;
            }
            ret = F.acc;
        }
        if (top == 0) return ret;
        top--;
        mode = 1;
    }
}

__device__ static int gateValue(const GfGate *G, uint64_t seed, const DevOct *o)
{
    Lattice lat;
    latStart(lat, seed);
    int acc = 1;
    for (int f = 0; f < G->nfilter; f++) {
        int v = treeValue(G, G->root[f], lat, o);
        if (v == 0) return 0;
        if (v == 2) acc = 2;
    }
    return acc;
}

struct GateArgs {
    const GfGate *G;            // NULL: crunch()
    uint64_t base;              // kGate: seeds base + i
    const uint64_t *seeds;      // kGateRules: the seeds
    int64_t n;
    uint8_t *flag;              // 0, 1, or 2 (needs the rules)
    const GfLayout *RL;         // kGateRules: the rule layout
};

// Every seed through the gate, leaving what only the sampled rules can decide.
__global__ void kGate(GateArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    uint64_t seed = A.base + (uint64_t)i;
    A.flag[i] = A.G ? (uint8_t)gateValue(A.G, seed, nullptr) : (crunchSeed(seed) ? 1 : 0);
}

// The undecided seeds again, with the rules' octaves held by the thread.
__global__ void kGateRules(GateArgs A)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A.n) return;
    uint64_t seed = A.seeds[i];
    DevOct o[GF_LOCALSLOT];
    initSeed(A.RL, seed, o);
    A.flag[i] = (uint8_t)gateValue(A.G, seed, o);
}

// ------------------------------------------------------------- compaction

struct CountArgs { const uint8_t *flag; int64_t n; uint8_t want; uint32_t *cnt; };

__global__ void kCount(CountArgs A)
{
    int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i0 = c * CHUNK;
    if (i0 >= A.n) return;
    int64_t i1 = i0 + CHUNK < A.n ? i0 + CHUNK : A.n;
    uint32_t k = 0;
    for (int64_t i = i0; i < i1; i++) k += A.flag[i] == A.want;
    A.cnt[c] = k;
}

struct WriteArgs {
    const uint8_t *flag;
    int64_t n;
    uint8_t want;
    uint64_t base;
    const uint64_t *seeds;
    const int64_t *offs;
    uint64_t *out;
};

__global__ void kWrite(WriteArgs A)
{
    int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i0 = c * CHUNK;
    if (i0 >= A.n) return;
    int64_t i1 = i0 + CHUNK < A.n ? i0 + CHUNK : A.n;
    int64_t at = A.offs[c];
    for (int64_t i = i0; i < i1; i++)
        if (A.flag[i] == A.want)
            A.out[at++] = A.seeds ? A.seeds[i] : A.base + (uint64_t)i;
}

// ------------------------------------------------------------------ state

static struct {
    int open;
    DevBuf tables, param, nodes, spline;
    DevBuf lte, lpre, lcoarse, lfull;       // layouts on the device
    GfLayout te, pre, coarse, full;         // and on the host
    DevBuf gate, lrule;
    int haveGate, gateRules;
    DevBuf flag, cnt, offs, oct, cells, arb, present, undecided;
    DevBuf list[2];                         // a gate's input and output
    // rows per launch: [gate], then the gate's rules pass, then fullscan cells
    int64_t pace[NSTAGE + 2];
    double cut[NSTAGE];
    int minBiomes;
    int32_t sampleRange, sampleStep, trueStep, sampleY;
    int setupDone;
} G;

enum { PACE_RULES = NSTAGE, PACE_CELLS = NSTAGE + 1 };

GF_API int gf_version(void) { return GF_VERSION; }
GF_API const char *gf_error(void) { return g_err; }

GF_API int gf_open(int device, char *msg, int msgLen)
{
    g_err[0] = 0;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count <= 0) {
        setErr("no CUDA device (%s)", e != cudaSuccess ? cudaGetErrorString(e) : "none found");
        if (msg) std::snprintf(msg, msgLen, "%s", g_err);
        return -1;
    }
    if (device < 0 || device >= count) {
        setErr("device %d does not exist (%d found)", device, count);
        if (msg) std::snprintf(msg, msgLen, "%s", g_err);
        return -1;
    }
    e = cudaSetDevice(device);
    if (e == cudaSuccess) e = cudaFree(nullptr);
    if (e != cudaSuccess) {
        setErr("could not open device %d: %s", device, cudaGetErrorString(e));
        if (msg) std::snprintf(msg, msgLen, "%s", g_err);
        return -1;
    }
    cudaDeviceProp p;
    std::memset(&p, 0, sizeof(p));
    cudaGetDeviceProperties(&p, device);
    if (msg) std::snprintf(msg, msgLen, "%s (compute %d.%d, %.1f GB)", p.name, p.major, p.minor,
                           p.totalGlobalMem / 1073741824.0);
    G.open = 1;
    // first launches small enough to be safe on a slow GPU; pace() grows them
    int64_t pace0[NSTAGE + 2] = { 1 << 22, 1 << 18, 1 << 13, 1 << 9, 1 << 12, 1, 1 << 12, 1 << 17 };
    for (int k = 0; k < NSTAGE + 2; k++) G.pace[k] = pace0[k];
    return 0;
}

GF_API void gf_close(void)
{
    G.tables.free(); G.param.free(); G.nodes.free(); G.spline.free();
    G.lte.free(); G.lpre.free(); G.lcoarse.free(); G.lfull.free();
    G.gate.free(); G.lrule.free();
    G.flag.free(); G.cnt.free(); G.offs.free(); G.oct.free();
    G.cells.free(); G.arb.free(); G.present.free(); G.undecided.free();
    G.list[0].free(); G.list[1].free();
    G.open = 0;
    G.setupDone = 0;
    G.haveGate = 0;
    G.gateRules = 0;
}

static int needOpen(void)
{
    if (!G.open) { setErr("gf_open has not been called"); return -1; }
    return 0;
}

static uint64_t usableBytes(void)
{
    size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) != cudaSuccess) return 0;
    uint64_t margin = 512ULL << 20;
    return f > margin ? f - margin : 0;
}

static int layoutValid(const GfLayout *L)
{
    if (L->nstream < 1 || L->nstream > GF_MAXSTREAM || L->nslot < 1 || L->nslot > GF_MAXSLOT) {
        setErr("layout has %d streams and %d slots", L->nstream, L->nslot);
        return -1;
    }
    for (int s = 0; s < L->nstream; s++) {
        const GfStream &S = L->stream[s];
        if (S.nA < 0 || S.nB < 0 || S.slotA < 0 || S.slotB < 0 ||
            S.slotA + S.nA > L->nslot || S.slotB + S.nB > L->nslot) {
            setErr("stream %d refers to slots outside the layout", s);
            return -1;
        }
    }
    for (int k = 0; k < L->nslot; k++)
        if (L->slot[k].stream < 0 || L->slot[k].stream >= L->nstream || (L->slot[k].half & ~1)) {
            setErr("slot %d is malformed", k);
            return -1;
        }
    return 0;
}

GF_API int gf_setup(const GfSetup *S)
{
    if (needOpen()) return -1;
    if (layoutValid(&S->te) || layoutValid(&S->pre) || layoutValid(&S->coarse) || layoutValid(&S->full)) return -1;
    if (S->te.nslot > GF_LOCALSLOT || S->pre.nslot > GF_LOCALSLOT || S->coarse.nslot > GF_LOCALSLOT) {
        setErr("the temp_even, prefilter and coarse layouts hold at most %d octaves", GF_LOCALSLOT);
        return -1;
    }
    if (S->nsteps < 1 || S->nsteps > 16 || S->len < 1 || S->len > 65535 || S->nspline < 1 ||
        S->nspline > GF_MAXSPLINE || S->numBiomes < 1 || S->numBiomes > GF_MAXBIOME) {
        setErr("biome tables out of range");
        return -1;
    }
    if (S->full.nstream < 6 || S->coarse.nstream < 5 || S->pre.nstream < 5) {
        setErr("a layout lacks a climate noise");
        return -1;
    }
    if (G.param.upload(S->param, 8 * (size_t)S->nparam) || G.nodes.upload(S->nodes, 8 * (size_t)S->len) ||
        G.spline.upload(S->spline, sizeof(GfSpline) * (size_t)S->nspline))
        return -1;
    DevTables T;
    std::memset(&T, 0, sizeof(T));
    T.order = S->order; T.len = S->len; T.nsteps = S->nsteps; T.nparam = S->nparam;
    for (int k = 0; k < 16; k++) T.steps[k] = k < S->nsteps ? S->steps[k] : 0;
    T.param = G.param.as<int32_t>();
    T.nodes = G.nodes.as<uint64_t>();
    T.spline = G.spline.as<GfSpline>();
    T.nspline = S->nspline;
    T.splineRoot = S->splineRoot;
    std::memcpy(T.idToIndex, S->idToIndex, sizeof(T.idToIndex));
    T.numBiomes = S->numBiomes;
    T.sampleRange = S->sampleRange; T.teStride = S->teStride; T.preStride = S->preStride;
    T.coarseScale = S->coarseScale; T.sampleY = S->sampleY; T.sampleStep = S->sampleStep; T.trueStep = S->trueStep;
    for (int k = 0; k < 4; k++) T.teBand[k] = S->teBand[k];
    for (int k = 0; k < 5; k++) T.teWeight[k] = S->teWeight[k];
    if (G.tables.upload(&T, sizeof(T))) return -1;
    G.te = S->te; G.pre = S->pre; G.coarse = S->coarse; G.full = S->full;
    if (G.lte.upload(&S->te, sizeof(GfLayout)) || G.lpre.upload(&S->pre, sizeof(GfLayout)) ||
        G.lcoarse.upload(&S->coarse, sizeof(GfLayout)) || G.lfull.upload(&S->full, sizeof(GfLayout)))
        return -1;
    G.cut[ST_TE] = S->teCut;
    G.cut[ST_PRE] = S->preCut;
    G.cut[ST_COARSE] = S->coarseCut;
    G.cut[ST_GRID] = S->gridCut;
    G.cut[ST_FULL] = S->fullCut;
    G.minBiomes = S->minBiomes;
    G.sampleRange = S->sampleRange;
    G.sampleStep = S->sampleStep;
    G.trueStep = S->trueStep;
    G.sampleY = S->sampleY;
    G.setupDone = 1;
    return 0;
}

GF_API int gf_gate(const GfGate *Gt)
{
    if (needOpen()) return -1;
    if (!Gt) { G.haveGate = 0; G.gateRules = 0; return 0; }
    if (Gt->nfilter < 1 || Gt->nfilter > GF_MAXFILTER || Gt->nnode < 1 || Gt->nnode > GF_MAXNODE ||
        Gt->natom < 0 || Gt->natom > GF_MAXATOM || Gt->nrule < 0 || Gt->nrule > GF_MAXRULE) {
        setErr("gate program out of range");
        return -1;
    }
    for (int k = 0; k < Gt->nnode; k++) {
        const GfNode &N = Gt->node[k];
        if (N.type == GF_N_LEAF && (N.atom < 0 || N.atom >= Gt->natom)) { setErr("node %d has no atom", k); return -1; }
        if ((N.type == GF_N_AND || N.type == GF_N_OR) && (N.nkid < 1 || N.nkid > 4)) { setErr("node %d has %d children", k, N.nkid); return -1; }
        for (int q = 0; q < N.nkid; q++)
            if (N.kid[q] < 0 || N.kid[q] >= Gt->nnode) { setErr("node %d has a bad child", k); return -1; }
    }
    for (int f = 0; f < Gt->nfilter; f++)
        if (Gt->root[f] < 0 || Gt->root[f] >= Gt->nnode) { setErr("filter %d has no root", f); return -1; }
    for (int a = 0; a < Gt->natom; a++)
        if (Gt->atom[a].kind == GF_A_RULE && (Gt->atom[a].rule < 0 || Gt->atom[a].rule >= Gt->nrule)) {
            setErr("atom %d has no rule", a);
            return -1;
        }
    if (Gt->nrule) {
        if (layoutValid(&Gt->ruleLayout)) return -1;
        if (Gt->ruleLayout.nslot > GF_LOCALSLOT) { setErr("the rules read more than %d octaves", GF_LOCALSLOT); return -1; }
    }
    if (G.gate.upload(Gt, sizeof(GfGate))) return -1;
    if (Gt->nrule && G.lrule.upload(&Gt->ruleLayout, sizeof(GfLayout))) return -1;
    G.haveGate = 1;
    G.gateRules = Gt->nrule > 0;
    return 0;
}

// ------------------------------------------------------------ batch pieces

// Adjusts a gate's rows per launch toward 0.3 - 1.2 s a launch. A launch is
// never shorter than its longest thread - a coarse seed is one thread walking
// all its cells - so launches are grown until they are that long and use every
// core, and only shrunk when they approach Windows' two-second limit.
static void pace(int stage, double ms, int64_t rows)
{
    int64_t &p = G.pace[stage];
    if (rows < p) return;
    if (ms < 300 && p < (1LL << 26)) p *= 2;
    else if (ms > 1200 && p > 1) p /= 2;
}

// Seeds where flag == want, in order, into out (grown as needed). Returns
// the count, or -1.
static int64_t compact(DevBuf &flag, int64_t n, uint8_t want, uint64_t base, const uint64_t *seedsIn, DevBuf &out)
{
    if (n <= 0) return 0;
    int64_t nchunk = (n + CHUNK - 1) / CHUNK;
    if (G.cnt.ensure((size_t)nchunk * 4) || G.offs.ensure((size_t)nchunk * 8)) return -1;
    CountArgs C = { flag.as<uint8_t>(), n, want, G.cnt.as<uint32_t>() };
    kCount<<<blocksFor(nchunk), BLOCK>>>(C);
    if (cudaGetLastError() != cudaSuccess) { setErr("count launch failed"); return -1; }
    uint32_t *hc = new uint32_t[nchunk];
    int64_t *ho = new int64_t[nchunk];
    cudaError_t e = cudaMemcpy(hc, G.cnt.p, (size_t)nchunk * 4, cudaMemcpyDeviceToHost);
    int64_t total = 0;
    for (int64_t c = 0; c < nchunk; c++) { ho[c] = total; total += hc[c]; }
    if (e == cudaSuccess) e = cudaMemcpy(G.offs.p, ho, (size_t)nchunk * 8, cudaMemcpyHostToDevice);
    delete[] hc;
    delete[] ho;
    if (e != cudaSuccess) { setErr("CUDA %s while compacting", cudaGetErrorString(e)); return -1; }
    if (!total) return 0;
    if (out.ensure((size_t)total * 8)) return -1;
    WriteArgs W = { flag.as<uint8_t>(), n, want, base, seedsIn, G.offs.as<int64_t>(), out.as<uint64_t>() };
    kWrite<<<blocksFor(nchunk), BLOCK>>>(W);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        setErr("write launch failed");
        return -1;
    }
    return total;
}

// temp_even, prefilter or coarse for seeds[0..n) (a device list): pass flags
// into G.flag.
static int stageFlags(int stage, DevBuf &layout, const GfLayout &L, const uint64_t *seeds, int64_t n)
{
    if (G.flag.ensure((size_t)(n ? n : 1))) return -1;
    for (int64_t at = 0; at < n;) {
        int64_t m = G.pace[stage];
        if (m > n - at) m = n - at;
        double t0 = nowMs();
        StageArgs A = { stage, layout.as<GfLayout>(), G.tables.as<DevTables>(), seeds + at, m, G.cut[stage],
                        G.flag.as<uint8_t>() + at };
        if (stage == ST_COARSE) kCoarse<<<blocksFor(m), BLOCK>>>(A);
        else if (L.nslot <= SMALL_SLOTS) kTePre<<<blocksFor(m), BLOCK>>>(A);
        else kTePreLarge<<<blocksFor(m), BLOCK>>>(A);
        LAUNCHED();
        TRY(cudaDeviceSynchronize());
        pace(stage, nowMs() - t0, m);
        at += m;
    }
    return 0;
}

// Octaves of a layout for seeds[0..n) into G.oct, as rows.
static int initRows(DevBuf &layout, const GfLayout &L, const uint64_t *seeds, int64_t n)
{
    if (G.oct.ensure(sizeof(DevOct) * (size_t)L.nslot * (size_t)n)) return -1;
    InitArgs I = { layout.as<GfLayout>(), seeds, n, G.oct.as<DevOct>() };
    kInit<<<blocksFor(n), BLOCK>>>(I);
    LAUNCHED();
    return 0;
}

// Scale-4 cells for seeds[0..n) (a device list), W x W of them from x0 in steps
// of dx, reduced per seed: values into G.arb, biome counts into G.present.
//   gridscan  the SAMPLE_STEP grid: x = bx/4 for bx = -range/2 + i*step
//   fullscan  TRUE_W x TRUE_W cells from -(range/2)/4
static int cellValues(int stage, const uint64_t *seeds, int64_t n, int32_t W, int32_t x0, int32_t dx)
{
    if (!n) return 0;
    if (G.arb.ensure(8 * (size_t)n) || G.present.ensure(4 * (size_t)n)) return -1;
    const GfLayout &L = G.full;
    int32_t y = G.sampleY / 4;
    int64_t cells = (int64_t)W * W;
    uint64_t perSeed = sizeof(DevOct) * (uint64_t)L.nslot + (uint64_t)cells;
    int64_t fit = (int64_t)(usableBytes() / 2 / perSeed);
    if (fit < 1) { setErr("no GPU memory for one seed's fullscan"); return -1; }
    for (int64_t at = 0; at < n;) {
        int64_t m = G.pace[stage];
        if (m > fit) m = fit;
        if (m > n - at) m = n - at;
        if (G.cells.ensure((size_t)(m * cells))) return -1;
        double tp = nowMs();
        if (initRows(G.lfull, L, seeds + at, m)) return -1;
        for (int64_t c0 = 0; c0 < cells;) {
            int64_t per = G.pace[PACE_CELLS] / m;
            if (per < 1) per = 1;
            if (per > cells - c0) per = cells - c0;
            double t0 = nowMs();
            CellArgs C = { G.lfull.as<GfLayout>(), G.oct.as<DevOct>(), G.tables.as<DevTables>(), m, c0, per,
                           W, x0, x0, dx, y, G.cells.as<int8_t>() };
            kCells<<<blocksFor(m * per), BLOCK>>>(C);
            LAUNCHED();
            TRY(cudaDeviceSynchronize());
            pace(PACE_CELLS, nowMs() - t0, m * per);
            c0 += per;
        }
        CellReduceArgs R = { G.tables.as<DevTables>(), G.cells.as<int8_t>(), m, cells,
                             G.arb.as<double>() + at, G.present.as<int32_t>() + at };
        kCellReduce<<<blocksFor(m), BLOCK>>>(R);
        LAUNCHED();
        TRY(cudaDeviceSynchronize());
        pace(stage, nowMs() - tp, m);
        at += m;
    }
    return 0;
}

static int gridValues(const uint64_t *seeds, int64_t n)
{
    int32_t W = G.sampleRange / G.sampleStep + 1;
    return cellValues(ST_GRID, seeds, n, W, (-(G.sampleRange / 2)) / 4, G.sampleStep / 4);
}

static int fullValues(const uint64_t *seeds, int64_t n)
{
    int32_t W = G.sampleRange / G.trueStep + 1;
    return cellValues(ST_FULL, seeds, n, W, -(G.sampleRange / 2) / 4, 1);
}

// dst[0..n1) followed by src[0..n2), in dst.
static int appendList(DevBuf &dst, int64_t n1, DevBuf &src, int64_t n2)
{
    if (n2 <= 0) return 0;
    DevBuf both;
    if (both.alloc(8 * (size_t)(n1 + n2))) return -1;
    if (n1) TRY(cudaMemcpy(both.p, dst.p, 8 * (size_t)n1, cudaMemcpyDeviceToDevice));
    TRY(cudaMemcpy(both.as<uint64_t>() + n1, src.p, 8 * (size_t)n2, cudaMemcpyDeviceToDevice));
    std::swap(dst.p, both.p);
    std::swap(dst.n, both.n);
    return 0;
}

// First-gate flags into G.flag: for seeds base .. base+n-1, or with rules set,
// for the n undecided seeds in G.undecided.
static int gateFlags(uint64_t base, int64_t n, int rules)
{
    if (G.flag.ensure((size_t)(n ? n : 1))) return -1;
    const GfGate *gp = G.haveGate ? G.gate.as<GfGate>() : nullptr;
    int st = rules ? PACE_RULES : ST_GATE;
    for (int64_t at = 0; at < n;) {
        int64_t m = G.pace[st];
        if (m > n - at) m = n - at;
        double t0 = nowMs();
        GateArgs A = { gp, base + (uint64_t)at, rules ? G.undecided.as<uint64_t>() + at : nullptr, m,
                       G.flag.as<uint8_t>() + at, G.lrule.as<GfLayout>() };
        if (rules) kGateRules<<<blocksFor(m), BLOCK>>>(A);
        else kGate<<<blocksFor(m), BLOCK>>>(A);
        LAUNCHED();
        TRY(cudaDeviceSynchronize());
        pace(st, nowMs() - t0, m);
        at += m;
    }
    return 0;
}

GF_API int gf_run(uint64_t base, int64_t n, uint64_t *out, int64_t cap, int64_t *nout, double *best, uint64_t *bestSeed)
{
    *nout = 0;
    *best = -1;
    *bestSeed = 0;
    if (needOpen()) return -1;
    if (!G.setupDone) { setErr("gf_setup has not been called"); return -1; }
    if (n <= 0) return 0;

    // the first gate over base .. base+n-1, then the seeds only its rules decide
    if (gateFlags(base, n, 0)) return -1;
    int64_t cur = compact(G.flag, n, 1, base, nullptr, G.list[0]);
    if (cur < 0) return -1;
    if (G.gateRules) {
        int64_t nu = compact(G.flag, n, 2, base, nullptr, G.undecided);
        if (nu < 0) return -1;
        if (nu > 0) {
            if (gateFlags(0, nu, 1)) return -1;
            int64_t n2 = compact(G.flag, nu, 1, 0, G.undecided.as<uint64_t>(), G.list[1]);
            if (n2 < 0 || appendList(G.list[0], cur, G.list[1], n2)) return -1;
            cur += n2;
        }
    }

    DevBuf *in = &G.list[0], *nx = &G.list[1];
    struct { int st; DevBuf *lay; const GfLayout *L; } plan[3] = {
        { ST_TE, &G.lte, &G.te }, { ST_PRE, &G.lpre, &G.pre }, { ST_COARSE, &G.lcoarse, &G.coarse },
    };
    for (int k = 0; k < 3 && cur > 0; k++) {
        if (stageFlags(plan[k].st, *plan[k].lay, *plan[k].L, in->as<uint64_t>(), cur)) return -1;
        int64_t got = compact(G.flag, cur, 1, 0, in->as<uint64_t>(), *nx);
        if (got < 0) return -1;
        cur = got;
        std::swap(in, nx);
    }
    if (cur <= 0) return 0;

    // gridscan: few seeds reach it, so its pass test is done here
    {
        if (gridValues(in->as<uint64_t>(), cur)) return -1;
        double *ha = new double[cur];
        uint8_t *hf = new uint8_t[cur];
        cudaError_t e = cudaMemcpy(ha, G.arb.p, 8 * (size_t)cur, cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < cur; i++) hf[i] = ha[i] >= G.cut[ST_GRID];
        if (e == cudaSuccess && G.flag.ensure((size_t)cur) == 0)
            e = cudaMemcpy(G.flag.p, hf, (size_t)cur, cudaMemcpyHostToDevice);
        delete[] ha;
        delete[] hf;
        TRY(e);
        int64_t got = compact(G.flag, cur, 1, 0, in->as<uint64_t>(), *nx);
        if (got < 0) return -1;
        cur = got;
        std::swap(in, nx);
    }
    if (cur <= 0) return 0;

    // fullscan, the same way
    if (fullValues(in->as<uint64_t>(), cur)) return -1;
    uint64_t *hs = new uint64_t[cur];
    double *ha = new double[cur];
    int32_t *hp = new int32_t[cur];
    cudaError_t e = cudaMemcpy(hs, in->p, 8 * (size_t)cur, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(ha, G.arb.p, 8 * (size_t)cur, cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(hp, G.present.p, 4 * (size_t)cur, cudaMemcpyDeviceToHost);
    int64_t passed = 0;
    if (e == cudaSuccess)
        for (int64_t i = 0; i < cur; i++) {
            if (hp[i] < G.minBiomes) continue;
            if (ha[i] > *best) { *best = ha[i]; *bestSeed = hs[i]; }
            if (ha[i] < G.cut[ST_FULL]) continue;
            if (passed < cap) out[passed] = hs[i];
            passed++;
        }
    delete[] hs;
    delete[] ha;
    delete[] hp;
    TRY(e);
    *nout = passed;
    return 0;
}
