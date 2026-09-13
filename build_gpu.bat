@echo off
rem Builds gpu_finder.dll from gpu_finder.cu with NVIDIA's nvcc, which needs
rem MSVC as its host compiler. `make gpu` runs this.
rem
rem CUDA:  CUDA_PATH if it is set, otherwise the newest toolkit unpacked under
rem        %LOCALAPPDATA%\cuda (the redistributable zips, no installer needed).
rem MSVC:  found through vswhere, from any Visual Studio with the C++ tools.
rem GPU:   compiled for the GPU in this machine. Set GPU_ARCH (e.g. sm_86) to
rem        target another.
rem
rem This folder's name holds an "&", so every path here stays inside quotes.
setlocal
set "HERE=%~dp0"

if not defined CUDA_PATH (
    for /d %%D in ("%LOCALAPPDATA%\cuda\*") do if exist "%%D\bin\nvcc.exe" set "CUDA_PATH=%%D"
)
if not defined CUDA_PATH goto nocuda
if not exist "%CUDA_PATH%\bin\nvcc.exe" goto nocuda

set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
set "VSDIR="
for /f "usebackq delims=" %%I in (`vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSDIR=%%I"
if not defined VSDIR goto nomsvc
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 goto nomsvc
set "PATH=%CUDA_PATH%\bin;%PATH%"

set "ARCH=%GPU_ARCH%"
if not defined ARCH set "ARCH=native"

cd /d "%HERE%"
echo nvcc: %CUDA_PATH%   arch: %ARCH%
nvcc -O3 -std=c++17 -arch=%ARCH% -shared gpu_finder.cu -o gpu_finder.dll
if errorlevel 1 exit /b 1
del /q gpu_finder.lib gpu_finder.exp 2>nul
echo built gpu_finder.dll
exit /b 0

:nocuda
echo CUDA toolkit not found. Set CUDA_PATH, or unpack NVIDIA's cuda_nvcc, libnvvm,
echo cuda_cudart, cuda_cccl and cuda_crt redistributable zips into
echo %%LOCALAPPDATA%%\cuda\^<version^>. The finder still runs on the CPU without it.
exit /b 1

:nomsvc
echo Visual Studio with the C++ tools was not found; nvcc needs MSVC on Windows.
exit /b 1
