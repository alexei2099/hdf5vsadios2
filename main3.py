import adios2
import numpy as np

# Define some data to write
data = np.random.rand(10)

# Step 1: Create an ADIOS object
with adios2.ADIOS() as adios_obj:
    # Step 2: Define an IO object and set up a variable
    io = adios_obj.DeclareIO("my_io")
    var = io.DefineVariable("my_data", data.shape, (0,), data.shape, adios2.ConstantDims, data)
    
    # Step 3: Open a file for writing
    # Note: 'w' mode for writing, 'r' for reading
    with io.Open("my_data.bp", adios2.Mode.Write) as engine:
        # Step 4: Write the data and close the file
        engine.Put(var, data)
        engine.PerformPuts()  # Make sure all data is written
        engine.Close()
