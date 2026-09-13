CC      = gcc
CFLAGS  = -O3 -march=native -flto -fwrapv -Wall -Ilib
LDFLAGS = -lm -lpthread

CORE_SRCS = lib/generator.c lib/layers.c lib/biomenoise.c lib/biomes.c lib/noise.c lib/terrainnoise.c
CORE_OBJS = $(CORE_SRCS:.c=.o)

ifeq ($(OS),Windows_NT)
    EXE   = .exe
    DLLIB =
else
    EXE   =
    DLLIB = -ldl
endif

# `make` builds the GPU library too. Without the CUDA toolkit that step fails
# (the leading '-' lets it) and main still runs on the CPU.
all: main$(EXE) gpu

cpu: main$(EXE)

main$(EXE): main.c filter.h gpu_host.c gpu_host.h gpu_finder.h $(CORE_OBJS)
	$(CC) $(CFLAGS) main.c gpu_host.c $(CORE_OBJS) -o $@ $(LDFLAGS) $(DLLIB)

gpu: gpu_finder.dll

# run from PowerShell or cmd; cmd does not search the current folder, hence .\\
gpu_finder.dll: gpu_finder.cu gpu_finder.h build_gpu.bat
	-cmd /c .\\build_gpu.bat

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: main$(EXE)
	./main$(EXE)

clean:
	rm -f main main.exe $(CORE_OBJS) gpu_finder.dll

.PHONY: all cpu gpu run clean
