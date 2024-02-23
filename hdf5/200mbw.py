import time
import numpy as np
import h5py

def generate_dataset():
    # Generate a dataset of approximately 200MB
    num_elements = (200 * 1024 * 1024) // np.dtype(np.float64).itemsize
    return np.random.rand(num_elements).astype(np.float64)

def write_hdf5(dataset, filename='dataset.h5'):
    start_time = time.time()
    with h5py.File(filename, 'w') as fw:
        fw.create_dataset('my_dataset', data=dataset)
    write_time = time.time() - start_time
    print(f"Write time with HDF5: {write_time} seconds.")
    # No direct method to get written data size from h5py, so use dataset.nbytes

def read_hdf5(filename='dataset.h5'):
    start_time = time.time()
    with h5py.File(filename, 'r') as fr:
        dataset_read = fr['my_dataset'][:]
    read_time = time.time() - start_time
    print(f"Read time with HDF5: {read_time} seconds.")
    return dataset_read, read_time

if __name__ == "__main__":
    dataset = generate_dataset()
    original_size_MB = dataset.nbytes / (1024 * 1024)
    print("Original dataset size:", original_size_MB, "MB")

    write_hdf5(dataset)
    dataset_read, read_time = read_hdf5()

    read_size_MB = dataset_read.nbytes / (1024 * 1024)
    print("Size of the dataset after reading:", read_size_MB, "MB")

    # Verify that the original dataset size is the same after writing
    assert original_size_MB == read_size_MB, "The dataset size after writing does not match the original."

    print("The dataset size is consistent before and after writing.")
