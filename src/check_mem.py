import os
import re

# Directory to search for files
directory = "/home/kausik/Documents/BundleTrack2.0/"

# Regular expression patterns to match cudaMalloc and cudaFree
cudaMalloc_pattern = r"cudaMalloc\(" #r"cuda_cache::alloc\("
# cudaMemset_pattern = r"cudaMemset\("
cudaFree_pattern = r"cudaFree\(" #r"cuda_cache::free\("

# Traverse all files in the directory
for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".c") or file.endswith(".cpp"):
                filepath = os.path.join(root, file)
                if "Thirdparty" not in filepath:
                    with open(filepath, "r") as f:
                        content = f.read()

                        # Count occurrences of cudaMalloc and cudaFree
                        num_cudaMalloc = len(re.findall(cudaMalloc_pattern, content))
                        # num_cudaMemset = len(re.findall(cudaMemset_pattern, content))
                        num_cudaFree = len(re.findall(cudaFree_pattern, content))

                        # Check if the counts match
                        if num_cudaMalloc != num_cudaFree:
                            print(f"In file {filepath}:")
                            print(f"  cudaMalloc count: {num_cudaMalloc}")
                            # print(f"  cudaMemset count: {num_cudaMemset}")
                            print(f"  cudaFree count: {num_cudaFree}")
                            print("  Missing cudaFree for some cudaMalloc calls")

