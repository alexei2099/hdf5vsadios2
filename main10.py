import time
import numpy as np
import adios2

def generate_dataset():
    # Generar un dataset de aproximadamente 100MB
    num_elements = (100 * 1024 * 1024) // np.dtype(np.float64).itemsize
    return np.random.rand(num_elements).astype(np.float64)

def write_adios(dataset, filename='dataset.bp'):
    shape = dataset.shape
    start_time = time.time()
    with adios2.open(filename, 'w') as fw:
        fw.write('my_dataset', dataset, shape, [0], shape)
    write_time = time.time() - start_time
    print(f"Tiempo de escritura con ADIOS2: {write_time} segundos.")
    return write_time  # Devolver el tiempo de escritura

def read_adios(filename='dataset.bp'):
    start_time = time.time()
    with adios2.open(filename, 'r') as fr:
        for step in fr:
            dataset_read = step.read('my_dataset')
    read_time = time.time() - start_time
    print(f"Tiempo de lectura con ADIOS2: {read_time} segundos.")
    return dataset_read, read_time

if __name__ == "__main__":
    dataset = generate_dataset()
    original_size = dataset.nbytes / (1024 * 1024)
    print("Tamaño del dataset original:", original_size, "MB")

    write_time = write_adios(dataset)
    print(f"Tiempo de escritura almacenado: {write_time} segundos.")

    dataset_read, read_time = read_adios()
    read_size = dataset_read.nbytes / (1024 * 1024)
    print("Tamaño del dataset después de leer:", read_size, "MB")

    # Verificar que el tamaño del dataset original es el mismo después de la escritura
    assert original_size == read_size, "El tamaño del dataset después de la escritura no coincide con el original."

    print("El tamaño del dataset es consistente antes y después de la escritura.")
