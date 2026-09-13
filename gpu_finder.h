// gpu_finder.h - the C interface of gpu_finder.dll, the finder's gates on the
// GPU. main.exe loads the DLL at run time (gpu_host.c), so it still runs, on
// the CPU, where the DLL or a CUDA device is missing.
//
// A batch of consecutive seeds goes through every gate in main.c's order -
// crunch() or the filters, temp_even, prefilter, coarse, gridscan, fullscan -
// each gate only seeing the seeds that passed the one before.

#ifndef GPU_FINDER_H
#define GPU_FINDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GF_VERSION     2
#define GF_MAXSTREAM   6
#define GF_MAXSLOT     64
#define GF_LOCALSLOT   40       // slots of the te, pre, coarse and rule layouts
#define GF_MAXTERM     16
#define GF_MAXATOM     128
#define GF_MAXNODE     256
#define GF_MAXFILTER   8
#define GF_MAXRULE     16
#define GF_MAXSPLINE   256
#define GF_MAXBIOME    64

// ---- octave layouts
// A stream is one noise: its md5 picks the noise's xoroshiro stream, whose
// first two longs seed half A and the next two half B. A slot is one perlin
// octave of one half.
typedef struct {
    uint64_t mlo, mhi;          // md5 "minecraft:<name>"
    double   amp;               // DoublePerlinNoise amplitude
    int32_t  slotA, nA;         // slots of half A, live octaves in order
    int32_t  slotB, nB;
} GfStream;

typedef struct {
    uint64_t mlo, mhi;          // md5 "octave_<n>" of this live octave
    double   amplitude;         // PerlinNoise.amplitude
    double   lacunarity;        // PerlinNoise.lacunarity
    int32_t  stream, half;
} GfSlot;

typedef struct {
    int32_t  nstream, nslot;
    GfStream stream[GF_MAXSTREAM];
    GfSlot   slot[GF_MAXSLOT];
} GfLayout;

// One node of the depth spline (initBiomeNoise), flattened: kid[] are indices
// into the same array. A node with len 1 is a fixed value.
typedef struct {
    int32_t len, typ;
    float   val;
    float   loc[12];
    float   der[12];
    int32_t kid[12];
} GfSpline;

// Stream order in every layout: temperature, humidity, continentalness,
// erosion, weirdness, then (full only) the shift noise.
typedef struct {
    // the biome search tree
    int32_t order, len, nsteps, nparam;
    uint32_t steps[16];
    const int32_t  *param;      // [nparam][2]
    const uint64_t *nodes;      // [len]
    int32_t nspline, splineRoot;
    const GfSpline *spline;
    int16_t idToIndex[256];     // biome id -> scored biome, or -1
    int32_t numBiomes;
    // main.c's #defines
    int32_t sampleRange, teStride, preStride, coarseScale, sampleY, sampleStep, trueStep;
    double  teBand[4], teWeight[5];
    // a seed passes a gate when its value is >= the cut; fullscan also needs
    // minBiomes biomes
    double  teCut, preCut, coarseCut, gridCut, fullCut;
    int32_t minBiomes;

    GfLayout te;                // temperature, TE_OCTAVE octaves
    GfLayout pre;               // the five climate noises, RED_OCTAVE octaves
    GfLayout coarse;            // the five, every live octave
    GfLayout full;              // the five and the shift noise, every live octave
} GfSetup;

// ---- the filters, as a program for the first gate
enum { GF_A_PHASE, GF_A_SUM, GF_A_RULE };
enum { GF_N_LEAF, GF_N_AND, GF_N_OR, GF_N_CRUNCH };
enum { GF_TK_BLOCK, GF_TK_FARNEAR, GF_TK_ORIGIN, GF_TK_PHASE };

// A phase is |frac(offset) - 0.5| of one lattice offset, oriented; a sum adds
// standardised phases. pi indexes the five climate noises in stream order;
// off is 0, 1, 2 for the a, b, c offsets.
typedef struct {
    int32_t kind, nsum;
    int32_t pi[4], half[4], oct[4], off[4];     // phase: [0]; sum: each part
    double  sign[4];                            // phase: [0]; sum: each part's orientation
    double  mu[4], sd[4];                       // sum parts
    int32_t rule, pad;                          // GF_A_RULE: which rule
    double  rsign;                              // GF_A_RULE: the atom's orientation
} GfAtom;

// A sampled rule. Block and far-near terms read octaves of stream[] in the
// gate's rule layout; origin and phase terms read lattice offsets directly:
// pi, bx = half, bz = octave, metric = offset.
typedef struct {
    int32_t sym, agg, nterm, pad;
    int32_t kind[GF_MAXTERM], stream[GF_MAXTERM], metric[GF_MAXTERM];
    int32_t bx[GF_MAXTERM], bz[GF_MAXTERM], pi[GF_MAXTERM];
    double  sign[GF_MAXTERM], mu[GF_MAXTERM], sd[GF_MAXTERM];
} GfRule;

// A tree node: a leaf passes when its atom's value is a number >= t.
typedef struct {
    int32_t type, nkid;
    int32_t kid[4];
    int32_t atom, pad;
    double  t;
} GfNode;

typedef struct {
    int32_t  nfilter;
    int32_t  root[GF_MAXFILTER];        // each filter's root node, applied in order
    int32_t  nnode, natom, nrule;
    // md5 of each climate noise, and of its first two live octaves
    uint64_t pmd5[5][2];
    uint64_t omd5[5][2][2];
    GfNode   node[GF_MAXNODE];
    GfAtom   atom[GF_MAXATOM];
    GfRule   rule[GF_MAXRULE];
    GfLayout ruleLayout;
    int32_t  farnear[32];               // near x, near z, far x, far z (8 each)
} GfGate;

#if defined(_WIN32) && defined(GF_BUILDING)
#define GF_API __declspec(dllexport)
#else
#define GF_API
#endif

GF_API int  gf_version(void);
// Opens a device. msg receives its name, or why it could not be opened.
GF_API int  gf_open(int device, char *msg, int msgLen);
GF_API void gf_close(void);
GF_API const char *gf_error(void);

GF_API int  gf_setup(const GfSetup *S);
// The first gate: NULL for crunch().
GF_API int  gf_gate(const GfGate *G);

// Seeds base .. base+n-1 through every gate. out receives the seeds that pass
// them all (at most cap; *nout is how many passed). best / bestSeed: the
// highest fullscan value among seeds with at least minBiomes biomes (-1 when
// there are none).
GF_API int  gf_run(uint64_t base, int64_t n, uint64_t *out, int64_t cap, int64_t *nout,
                   double *best, uint64_t *bestSeed);

#ifdef __cplusplus
}
#endif

#endif
