# No logging available for Powershell

# Navigate to project directory
cd "C:\Users\User\Desktop\Not backed up\CUDA-Accelerated-Clustering-with-JAX"

# Clean and recreate build directory
if (Test-Path -Path build) {
    try {
        Remove-Item -Recurse -Force build
    } catch {
        Write-Host "Failed to delete 'build' directory. Attempting again with elevated permissions..."
        Start-Process powershell -ArgumentList "-Command Remove-Item -Recurse -Force build" -Verb RunAs
    }
}

New-Item -ItemType Directory -Name build
cd build

# Run CMake to configure the project
cmake -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2" ..

# Build the project
cmake --build . --config Release

# Run the program if build succeeds
if (Test-Path -Path ".\Release\kmeans_cuda.exe") {
    cd Release
    .\kmeans_cuda.exe
} else {
    Write-Host "Build failed or executable not found!"
}
cd ../..
