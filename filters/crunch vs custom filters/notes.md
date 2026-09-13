# Crunch
.\main.exe --count 100000000000
biome arbitrations finder | start 1000000000000000000 | 16 threads | cutoff 81.00
  GPU: NVIDIA GeForce RTX 4090 (compute 8.9, 24.0 GB) | first gate: crunch()
stopped at 1000000100000000000 | 100000.00M seeds in 1927s | hits=135 | best ARB=82.950 seed 1000000005897820295

14.27 seconds per 81+ seed
112 seeds 81-82 (17.21 seconds per)
23 seeds 82-83 (83.78 seconds per)

# Custom filters a & b
.\main.exe --count 100000000000
filter 1: cheap and[or;or] level 0.4 (a.filter)
filter 2: main2 and[or;or] level 0.995 (b.filter)
biome arbitrations finder | start 1000000000000000000 | 16 threads | cutoff 81.00
  GPU: NVIDIA GeForce RTX 4090 (compute 8.9, 24.0 GB) | first gate: filters/order.txt
stopped at 1000000100000000000 | 100000.00M seeds in 3568s | hits=182 | best ARB=84.365 seed 1000000008383298422

19.60 seconds per 81+ seed
152 seeds 81-82 (23.47 seconds per)
27 seeds 82-83 (132.15 seconds per)
2 seeds 83-84 (1784 seconds per)
1 seed 84-85 (3568 seconds per)

# More notes
These are possibly not statistically significant amounts of numbers, currently running through the first 1 trillion on each filter to see if these numbers hold up.