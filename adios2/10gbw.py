import time
import numpy as np
import adios2

def generate_dataset():
    # Generate a dataset of approximately 10GB
    num_elements = (10 * 1024 * 1024 * 1024) // np.dtype(np.float64).itemsize
    return np.random.rand(num_elements).astype(np.float64)

def write_adios(dataset, filename='dataset.bp'):
    shape = dataset.shape
    start_time = time.time()
    with adios2.open(filename, 'w') as fw:
        fw.write('my_dataset', dataset, shape, [0], shape)
    write_time = time.time() - start_time
    print(f"ADIOS2 write time: {write_time} seconds.")
    return write_time

def read_adios(filename='dataset.bp'):
    start_time = time.time()
    with adios2.open(filename, 'r') as fr:
        for step in fr:
            dataset_read = step.read('my_dataset')
    read_time = time.time() - start_time
    print(f"ADIOS2 read time: {read_time} seconds.")
    return dataset_read, read_time

if __name__ == "__main__":
    dataset = generate_dataset()
    original_size = dataset.nbytes / (1024 * 1024)
    print("Original dataset size:", original_size, "MB")

    write_time = write_adios(dataset)
    print(f"Stored write time: {write_time} seconds.")

    dataset_read, read_time = read_adios()
    read_size = dataset_read.nbytes / (1024 * 1024)
    print("Dataset size after reading:", read_size, "MB")

    assert original_size == read_size, "The dataset size after writing does not match the original."
    print("The dataset size is consistent before and after writing.")
