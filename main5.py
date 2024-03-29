import time
import h5py
import numpy as np
import adios2 as ad2

def generate_dataset():
    # Calculate the number of elements for at least 100MB of data
    num_elements = (100 * 2**20) // 8  # 100MB, with each float64 being 8 bytes
    dimension = int(np.sqrt(num_elements))  # Square root to get dimensions for a square array
    return np.random.rand(dimension, dimension)  # Adjusted to ensure the dataset is at least 100MB

def print_dataset_size(dataset):
    print("Dataset size:", dataset.nbytes / (2**20), "MB")

def write_hdf5(dataset):
    with h5py.File('test_hdf5.h5', 'w') as f:
        f.create_dataset('my_dataset', data=dataset)

def write_adios(dataset):
    with ad2.open('test_adios2.bp', 'w') as fw:
        fw.write('my_dataset', dataset)

def read_hdf5():
    with h5py.File('test_hdf5.h5', 'r') as f:
        dataset = f['my_dataset'][:]
    return dataset

def read_adios():
    with ad2.open('test_adios2.bp', 'r') as fr:
        dataset = fr.read('my_dataset')
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset()

    print_dataset_size(dataset)

    start_time = time.time()
    write_hdf5(dataset)
    hdf5_write_time = time.time() - start_time
    print("HDF5 Write Time: {:.3f} seconds".format(hdf5_write_time))

    start_time = time.time()
    hdf5_dataset = read_hdf5()
    hdf5_read_time = time.time() - start_time
    print("HDF5 Read Time: {:.3f} seconds".format(hdf5_read_time))

    start_time = time.time()
    write_adios(dataset)
    adios_write_time = time.time() - start_time
    print("ADIOS Write Time: {:.3f} seconds".format(adios_write_time))

    start_time = time.time()
    adios_dataset = read_adios()
    adios_read_time = time.time() - start_time
    print("ADIOS Read Time: {:.3f} seconds".format(adios_read_time))

    # Verify data consistency (optional)
    assert np.array_equal(dataset, hdf5_dataset), "HDF5 data mismatch"
    assert np.array_equal(dataset, adios_dataset), "ADIOS data mismatch"
