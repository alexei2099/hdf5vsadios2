import adios2

def write_adios():
    # Define the file name for output
    output_file = 'output.bp'

    # Using ADIOS2 in write mode
    with adios2.open(output_file, "w") as adios_file:
        # Create some data to write
        data = [1, 2, 3, 4, 5]
        
        # Write the data to the file
        adios_file.write("myData", data)

def read_adios(file_name):
    # Using ADIOS2 in read mode
    with adios2.open(file_name, "r") as adios_file:
        # Reading the data back
        for variable_name, data in adios_file:
            print(f"Variable: {variable_name}, Data: {data}")

if __name__ == "__main__":
    write_adios()  # Write data to an ADIOS file
    read_adios('output.bp')  # Read the data back from the file
