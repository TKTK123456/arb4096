// gpu_host.h - loads gpu_finder.dll and builds the tables it needs.
//
// main.c includes this and never windows.h, whose near/far macros clash with
// ordinary code.

#ifndef GPU_HOST_H
#define GPU_HOST_H

#include <stddef.h>
#include <stdint.h>

#include "biomenoise.h"
#include "gpu_finder.h"

typedef struct {
    int  (*version)(void);
    int  (*open)(int device, char *msg, int msgLen);
    void (*close)(void);
    const char *(*error)(void);
    int  (*setup)(const GfSetup *S);
    int  (*gate)(const GfGate *G);
    int  (*run)(uint64_t base, int64_t n, uint64_t *out, int64_t cap, int64_t *nout, double *best, uint64_t *bestSeed);
} GpuApi;

extern GpuApi GPU;

// Loads gpu_finder.dll from beside the program and opens a device. msg
// receives the device's name, or why the GPU cannot be used.
int  gpuOpen(int device, char *msg, size_t msgLen);

// Adds one noise to a layout with nA / nB octaves per half (99: every live
// octave). np is NP_* (NP_SHIFT for the shift noise). Returns the stream's
// index, or -1.
int  gpuAddStream(GfLayout *L, int np, int nA, int nB);

// The depth spline of a generator set up by initBiomeNoise, as index-linked
// nodes. Returns the node count, or -1 when it does not fit in max.
int  gpuFlattenSpline(const BiomeNoise *bn, GfSpline *out, int max, int *root);

#endif
