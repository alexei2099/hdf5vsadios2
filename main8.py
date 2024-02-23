import time
import h5py
import numpy as np
import adios2

def generate_dataset():
    # Asegurándonos de que el dataset es significativamente grande, por ejemplo, 100MB.
    num_elements = (100 * 2**20) // 8  # 100MB con cada float64 de 8 bytes
    dimension = int(np.sqrt(num_elements))  # Para un array cuadrado
    return np.random.rand(dimension, dimension)

def write_hdf5(dataset):
    start_time = time.time()
    with h5py.File('test_hdf5.h5', 'w') as f:
        f.create_dataset('my_dataset', data=dataset)
    return time.time() - start_time

def read_hdf5():
    start_time = time.time()
    with h5py.File('test_hdf5.h5', 'r') as f:
        data = f['my_dataset'][:]
    return time.time() - start_time, data

def write_adios(dataset):
    start_time = time.time()
    with adios2.open('test_adios2.bp', 'w') as fw:
        fw.write('my_dataset', dataset)
    return time.time() - start_time

def read_adios():
    start_time = time.time()
    with adios2.open('test_adios2.bp', 'r') as fr:
        var_info = fr.available_variables()
        assert 'my_dataset' in var_info, "Variable 'my_dataset' no encontrada."
        shape_str = var_info['my_dataset']['Shape']
        shape = tuple(map(int, shape_str.split(',')))
        data = np.zeros(shape, dtype=np.float64)
        fr.read('my_dataset', data, start=[0, 0], count=shape)
    return time.time() - start_time, data

if __name__ == "__main__":
    dataset = generate_dataset()
    print("Tamaño del dataset:", dataset.nbytes / (2**20), "MB")

    hdf5_write_time = write_hdf5(dataset)
    print("Tiempo de escritura HDF5:", hdf5_write_time, "segundos")

    adios_write_time = write_adios(dataset)
    print("Tiempo de escritura ADIOS2:", adios_write_time, "segundos")

    hdf5_read_time, hdf5_data = read_hdf5()
    print("Tiempo de lectura HDF5:", hdf5_read_time, "segundos")

    adios_read_time, adios_data = read_adios()
    print("Tiempo de lectura ADIOS2:", adios_read_time, "segundos")

    # Verificación de la integridad de los datos
    assert np.array_equal(dataset, hdf5_data), "Los datos de HDF5 no coinciden."
    assert np.array_equal(dataset, adios_data), "Los datos de ADIOS2 no coinciden."
