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
    print("Tamaño del dataset original:", dataset.nbytes / (1024 * 1024), "MB")

    write_adios(dataset)
    dataset_read, read_time = read_adios()

    # Imprimir el tamaño del dataset leído
    if dataset_read is not None:
        print("Tamaño del dataset leído:", dataset_read.nbytes / (1024 * 1024), "MB")
    else:
        print("El dataset leído está vacío o no se pudo leer correctamente.")

    # Verificación de consistencia de datos (opcional)
    assert np.array_equal(dataset, dataset_read), "Los datos leídos no coinciden con los datos escritos."
