import time
import h5py
import numpy as np
import adios2

def generate_dataset():
    num_elements = (100 * 2**20) // 8  # 100MB, with each float64 being 8 bytes
    dimension = int(np.sqrt(num_elements))  # To create a square array
    return np.random.rand(dimension, dimension)  # Dataset is at least 100MB

def print_dataset_size(dataset):
    print("Dataset size:", dataset.nbytes / (2**20), "MB")

def write_hdf5(dataset):
    with h5py.File('test_hdf5.h5', 'w') as f:
        f.create_dataset('my_dataset', data=dataset)

def write_adios(dataset):
    with adios2.open('test_adios2.bp', 'w') as fw:
        fw.write('my_dataset', dataset)

def read_hdf5():
    with h5py.File('test_hdf5.h5', 'r') as f:
        dataset = f['my_dataset'][:]
    return dataset

def read_adios():
    with adios2.open('test_adios2.bp', 'r') as fr:
        var_info = fr.available_variables()
        if 'my_dataset' not in var_info:
            raise KeyError("Variable 'my_dataset' not found in ADIOS2 file.")
        shape_str = var_info['my_dataset']['Shape']
        dataset_shape = tuple(map(int, shape_str.split(',')))
        dataset = np.zeros(dataset_shape, dtype=np.float64)
        fr.read('my_dataset', dataset)
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset()

    print_dataset_size(dataset)

    start_time = time.time()
    write_hdf5(dataset)
    hdf5_write_time = time.time() - start_time
    print(f"HDF5 Write Time: {hdf5_write_time} seconds")

    start_time = time.time()
    write_adios(dataset)
    adios_write_time = time.time() - start_time
    print(f"ADIOS Write Time: {adios_write_time} seconds")

    start_time = time.time()
    hdf5_dataset = read_hdf5()
    hdf5_read_time = time.time() - start_time
    print(f"HDF5 Read Time: {hdf5_read_time} seconds")

    start_time = time.time()
    adios_dataset = read_adios()
    adios_read_time = time.time() - start_time
    print(f"ADIOS Read Time: {adios_read_time} seconds")

    # Data consistency checks
    assert np.array_equal(dataset, hdf5_dataset), "HDF5 data mismatch"
    assert np.array_equal(dataset, adios_dataset), "ADIOS data mismatch"
