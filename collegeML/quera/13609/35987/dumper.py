import zlib
import zipfile

import numpy as np
np.savez('data.npz',ethnics=ethnics,balance=balance,allow_pickle=True)

def compress(file_names):
    print("File Paths:")
    print(file_names)

    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    with zipfile.ZipFile("result.zip", mode="w") as zf:
	    for file_name in file_names:
	        # Add file to the zip file
	        # first parameter file to zip, second filename in zip
	        zf.write('./'+file_name, file_name, compress_type=compression)


file_names= ["data.npz", "solution.ipynb"]
compress(file_names)