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
        fw.write('my_dataset', dataset, dataset.shape, (0,0), dataset.shape, ad2.ConstantDims)

def read_hdf5():
    with h5py.File('test_hdf5.h5', 'r') as f:
        dataset = f['my_dataset'][:]
    return dataset

def read_adios():
    dataset = None
    with ad2.open('test_adios2.bp', 'r') as fr:
        for step in fr:
            vars_info = step.available_variables()
            shape = tuple(map(int, vars_info['my_dataset']['Shape'].split(',')))
            dataset = np.zeros(shape)
            step.read('my_dataset', dataset, ad2.Selection(slice(None)))
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset()

    print_dataset_size(dataset)

    start_time = time.time()
    write_hdf5(dataset)
    hdf5_write_time = time.time() - start_time
    print("HDF5 Write Time:", hdf5_write_time)

    start_time = time.time()
    write_adios(dataset)
    adios_write_time = time.time() - start_time
    print("ADIOS Write Time:", adios_write_time)

    start_time = time.time()
    hdf5_dataset = read_hdf5()
    hdf5_read_time = time.time() - start_time
    print("HDF5 Read Time:", hdf5_read_time)

    start_time = time.time()
    adios_dataset = read_adios()
    adios_read_time = time.time() - start_time
    print("ADIOS Read Time:", adios_read_time)

    # Verify data consistency (optional)
    assert np.array_equal(dataset, hdf5_dataset), "HDF5 data mismatch"
    assert np.array_equal(dataset, adios_dataset), "ADIOS data mismatch"
