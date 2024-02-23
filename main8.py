import time
import h5py
import numpy as np
import adios2

def generate_dataset():
    # Ajustamos la generación del dataset para asegurar que sea de al menos 100MB
    num_elements = (100 * 2**20) // 8  # 100MB, con cada float64 siendo 8 bytes
    dimension = int(np.sqrt(num_elements))  # Raíz cuadrada para obtener dimensiones de un array cuadrado
    return np.random.rand(dimension, dimension)

def print_dataset_size(dataset):
    print("Tamaño del dataset:", dataset.nbytes / (2**20), "MB")

def write_hdf5(dataset):
    with h5py.File('test_hdf5.h5', 'w') as f:
        f.create_dataset('my_dataset', data=dataset)

def write_adios(dataset):
    with adios2.open('test_adios2.bp', 'w') as fw:
        try:
            fw.write('my_dataset', dataset)
            print("Escritura ADIOS2 completada con éxito.")
        except Exception as e:
            print("Error durante la escritura ADIOS2:", e)

def read_hdf5():
    with h5py.File('test_hdf5.h5', 'r') as f:
        dataset = f['my_dataset'][:]
    return dataset

def read_adios():
    with adios2.open('test_adios2.bp', 'r') as fr:
        variables_info = fr.available_variables()
        if 'my_dataset' in variables_info:
            dataset = fr.read('my_dataset')
            return dataset
        else:
            print("Variable 'my_dataset' no encontrada en el archivo ADIOS2.")
            return None

if __name__ == "__main__":
    dataset = generate_dataset()

    print_dataset_size(dataset)

    start_time = time.time()
    write_hdf5(dataset)
    hdf5_write_time = time.time() - start_time
    print("Tiempo de escritura HDF5:", hdf5_write_time, "segundos")

    start_time = time.time()
    write_adios(dataset)
    adios_write_time = time.time() - start_time
    print("Tiempo de escritura ADIOS2:", adios_write_time, "segundos")

    start_time = time.time()
    hdf5_dataset = read_hdf5()
    hdf5_read_time = time.time() - start_time
    print("Tiempo de lectura HDF5:", hdf5_read_time, "segundos")

    start_time = time.time()
    adios_dataset = read_adios()
    adios_read_time = time.time() - start_time
    print("Tiempo de lectura ADIOS2:", adios_read_time, "segundos")

    # Verificación de consistencia de datos (opcional)
    assert np.array_equal(dataset, hdf5_dataset), "Desajuste de datos HDF5"
    if adios_dataset is not None:
        assert np.array_equal(dataset, adios_dataset), "Desajuste de datos ADIOS2"
