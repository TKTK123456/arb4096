# Seed Finder GPU & Custom Filters

Scans consecutive Minecraft seeds for biome diversity in the 4096 x 4096
blocks around the origin, and records every seed that scores above
`SCORE_CUTOFF`. Runs on an NVIDIA GPU, or on the CPU.

## Requirements

- Windows, with gcc and make (MSYS2's MinGW-w64 toolchain).
- For the GPU:
  - an NVIDIA GPU from the GTX 16 / RTX 20 series or newer (compute
    capability 7.5+) and a current driver;
  - the CUDA Toolkit 13, found through `CUDA_PATH` or under
    `%LOCALAPPDATA%\cuda\<version>`;
  - Visual Studio with the C++ tools (CUDA's compiler needs it on Windows).

Without a usable GPU the finder says why and runs on the CPU.

## Building

```
make         # main.exe and gpu_finder.dll
make cpu     # main.exe only
```

`gpu_finder.dll` is compiled for the GPU in the machine that builds it. To
build for another GPU, set `GPU_ARCH` (for example `sm_86`) first. Stop a
running `main.exe` before rebuilding.

## Running

```
main                   # from the checkpoint, or START_SEED
main --start <seed>    # from <seed>
main --count <n>       # stop after <n> seeds
main single <seed>     # every gate's value for one seed
```

Hits are printed and appended to `arb4096_results.txt` as
`seed <tab> score <tab> biomes`. Progress is saved to `arb4096_checkpoint.txt`
every 30 seconds and on exit (Ctrl+C), and the next run resumes from it.

## Settings

Everything is set with the `#define`s at the top of `main.c`; rebuild after
changing them.

| define | |
| --- | --- |
| `USE_GPU` | `1` GPU (the default), `0` CPU |
| `USE_FILTERS` | `0` `crunch()` is the first gate, `1` the filters listed in the rule file |
| `FILTER_RULES` | the rule file, inside `filters/` |
| `SCORE_CUTOFF`, `REQUIRE_ALL_BIOMES` | what counts as a hit |
| `TE_CUTOFF`, `PRE_CUTOFF`, `COARSE_CUTOFF`, `CRUNCH_*` | how strict each gate is |
| `NUM_THREADS` | CPU threads |

## Filters

With `USE_FILTERS 1`, the seeds are first run through the `.filter` files
named in the rule file, instead of `crunch()`. The rule file lists one file
per line, and they apply top to bottom, each to the seeds the ones above it
kept. Blank lines and lines starting with `#` are ignored.

```
# filters/order.txt
a.filter
b.filter
```

## Notes

- Every seed the GPU finds is scored again on the CPU before it is recorded,
  so both backends record the same hits.
- On Ctrl+C the GPU stops at once, and the CPU finishes scoring what the GPU
  already found before the program exits.

## Files

| file | |
| --- | --- |
| `main.c` | the finder: settings, gates, CPU and GPU runs |
| `filter.h` | reads and evaluates `.filter` files |
| `gpu_finder.cu`, `gpu_finder.h` | the gates as CUDA kernels (`gpu_finder.dll`) |
| `gpu_host.c`, `gpu_host.h` | loads `gpu_finder.dll` and prepares its tables |
| `build_gpu.bat` | builds `gpu_finder.dll` (`make gpu` runs it) |
| `filters/` | `.filter` files and the rule file |
| `lib/` | the biome generation library |
