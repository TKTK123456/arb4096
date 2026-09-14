CC      = gcc
CFLAGS  = -O3 -march=native -flto -fwrapv -Wall -Ilib
LDFLAGS = -lm -lpthread

CORE_SRCS = lib/generator.c lib/layers.c lib/biomenoise.c lib/biomes.c lib/noise.c lib/terrainnoise.c
CORE_OBJS = $(CORE_SRCS:.c=.o)

ifeq ($(OS),Windows_NT)
    EXE    = .exe
    DLLIB  =
    GPULIB = gpu_finder.dll
else
    EXE    =
    DLLIB  = -ldl
    GPULIB = gpu_finder.so
endif

# CUDA compiler/flags used on Linux only
NVCC      = nvcc
NVCCFLAGS = -O3 --shared -Xcompiler -fPIC

# `make` builds the GPU library too. Without the CUDA toolkit that step fails
# (the leading '-' lets it) and main still runs on the CPU.
all: main$(EXE) gpu

cpu: main$(EXE)

main$(EXE): main.c filter.h gpu_host.c gpu_host.h gpu_finder.h $(CORE_OBJS)
	$(CC) $(CFLAGS) main.c gpu_host.c $(CORE_OBJS) -o $@ $(LDFLAGS) $(DLLIB)

gpu: $(GPULIB)

# run from PowerShell or cmd; cmd does not search the current folder, hence .\\
gpu_finder.dll: gpu_finder.cu gpu_finder.h build_gpu.bat
	-cmd /c .\\build_gpu.bat

# Linux CUDA build: produces a shared object instead of a .dll
gpu_finder.so: gpu_finder.cu gpu_finder.h
	-$(NVCC) $(NVCCFLAGS) gpu_finder.cu -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: main$(EXE)
	./main$(EXE)

clean:
	rm -f main main.exe $(CORE_OBJS) gpu_finder.dll gpu_finder.so

.PHONY: all cpu gpu run clean