#!/bin/bash

# Define project root directory as the current directory
project_dir="$(pwd)"
log_file="build_and_run.log"
temp_log_file="temp_build_and_run.log"

# Ensure the script is running in the project root
if [ ! -d "$project_dir" ]; then
    echo "Project directory not found: $project_dir"
    exit 1
fi

# Check if CUDAPATH is set
if [ -z "$CUDAPATH" ]; then
    echo "CUDAPATH environment variable is not set. Please set it to your CUDA installation path."
    exit 1
fi

# Remove existing log files
[ -f "$log_file" ] && rm "$log_file"
[ -f "$temp_log_file" ] && rm "$temp_log_file"

# Redirect all output to a temporary log file
exec > >(tee "$temp_log_file") 2>&1

# Start the script
echo "Starting build and run script in: $project_dir"

# Clean and recreate build directory
if [ -d build ]; then
    echo "Cleaning up existing build directory..."
    rm -rf build || { echo "Failed to delete 'build' directory"; exit 1; }
fi

mkdir build
cd build || { echo "Failed to create or navigate to 'build' directory"; exit 1; }

# Run CMake to configure the project
echo "Configuring the project with CMake..."
cmake -DCMAKE_GENERATOR_TOOLSET="cuda=$CUDAPATH" .. || { echo "CMake configuration failed"; exit 1; }

# Build the project
echo "Building the project..."
cmake --build . --config Release || { echo "Build failed"; exit 1; }

# Run the program if build succeeds
if [ -f "./Release/kmeans_cuda.exe" ]; then
    echo "Running the program..."
    cd Release || { echo "Failed to navigate to 'Release' directory"; exit 1; }
    ./kmeans_cuda.exe || { echo "Failed to run program"; exit 1; }
else
    echo "Build failed or executable not found!"
fi

# Return to project root
cd ../..

echo "Script completed successfully!"

# Basic sanitization for the log
echo "Sanitizing the log..."
sed -e "s|$CUDAPATH|<cuda-path>|g" \
    -e "s|C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools|<visual-studio-path>|g" \
    -e "s|C:/WINDOWS|<windows-dir>|g" \
    "$temp_log_file" > "$log_file"

# Clean up temporary log
rm "$temp_log_file"

echo "Sanitized log saved to: $log_file"
