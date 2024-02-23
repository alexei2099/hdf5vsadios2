import time
import numpy as np
import adios2

def generate_dataset():
    # Cada float64 ocupa 8 bytes, así que para 100MB necesitamos 12,500,000 elementos
    num_elements = (100 * 1024 * 1024) // 8
    # Creamos un array 1D por simplicidad, aunque podrías ajustarlo a otras formas si es necesario
    return np.random.rand(num_elements)

def write_adios(dataset, filename='dataset.bp'):
    with adios2.open(filename, 'w') as fw:
        start_time = time.time()
        fw.write('my_dataset', dataset)
        write_time = time.time() - start_time
        print(f"Tiempo de escritura con ADIOS2: {write_time} segundos.")
        return write_time

def read_adios(filename='dataset.bp'):
    with adios2.open(filename, 'r') as fr:
        start_time = time.time()
        dataset = fr.read('my_dataset')
        read_time = time.time() - start_time
        print(f"Tiempo de lectura con ADIOS2: {read_time} segundos.")
        return dataset, read_time

if __name__ == "__main__":
    dataset = generate_dataset()
    print("Tamaño del dataset:", dataset.nbytes / (1024 * 1024), "MB")
    
    write_time = write_adios(dataset)
    dataset_read, read_time = read_adios()

    # Verificación de consistencia de datos (opcional)
    assert np.array_equal(dataset, dataset_read), "Los datos leídos no coinciden con los datos escritos."
