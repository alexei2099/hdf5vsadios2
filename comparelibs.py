import time
import h5py
import numpy as np
import adios2

def generate_dataset():
    return np.random.rand(1000, 1000)  # Example dataset of shape (1000, 1000)

def write_hdf5(dataset):
    with h5py.File('test_hdf5.h5', 'w') as f:
        f.create_dataset('my_dataset', data=dataset)

def write_adios(dataset):
    adios = adios2.Adios
    io = adios.DeclareIO("myio")
    writer = io.Open("test_adios2.bp", adios2.Mode.Write)
    writer.Put("my_dataset", dataset)
    writer.Close()

def read_hdf5():
    with h5py.File('test_hdf5.h5', 'r') as f:
        dataset = f['my_dataset'][:]
    return dataset

def read_adios():
    adios = adios2.Adios
    io = adios.DeclareIO("myio")
    reader = io.Open("test_adios2.bp", adios2.Mode.Read)
    dataset = reader.Get("my_dataset")
    reader.Close()
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset()

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
    assert np.array_equal(dataset, hdf5_dataset)
    assert np.array_equal(dataset, adios_dataset)
