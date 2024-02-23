import numpy as np
import h5py
import time

# Function to generate dataset
def generate_dataset(size_mb=500):
    """Generate a random dataset of approximately size_mb MB"""
    num_elements = (size_mb * 1024 * 1024) // np.dtype(np.float64).itemsize
    return np.random.rand(num_elements).astype(np.float64)

# Function to write dataset using HDF5
def write_hdf5(dataset, filename='dataset_500mb.h5'):
    start_time = time.time()
    with h5py.File(filename, 'w') as fw:
        fw.create_dataset('my_dataset', data=dataset)
    write_time = time.time() - start_time
    print(f"Write time with HDF5: {write_time} seconds.")
    return write_time

# Function to read dataset using HDF5
def read_hdf5(filename='dataset_500mb.h5'):
    start_time = time.time()
    with h5py.File(filename, 'r') as fr:
        dataset_read = fr['my_dataset'][:]
    read_time = time.time() - start_time
    print(f"Read time with HDF5: {read_time} seconds.")
    return dataset_read, read_time

# Main script
if __name__ == "__main__":
    dataset = generate_dataset(500)  # Generate 500 MB dataset
    original_size_MB = dataset.nbytes / (1024 * 1024)
    print("Original dataset size: {:.2f} MB".format(original_size_MB))

    write_time = write_hdf5(dataset)
    dataset_read, read_time = read_hdf5()

    read_size_MB = dataset_read.nbytes / (1024 * 1024)
    print("Read dataset size: {:.2f} MB".format(read_size_MB))

    # Assertion to check if the written and read datasets are the same size
    assert original_size_MB == read_size_MB, "Dataset size mismatch after writing and reading."

    print("Dataset integrity verified: original and read dataset sizes match.")
