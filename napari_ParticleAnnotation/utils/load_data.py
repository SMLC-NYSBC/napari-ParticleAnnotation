import numpy as np
import mrcfile


def load_image(path):
    paths = [path] if isinstance(path, str) else path

    add_kwargs = {}

    # optional, default is "image"
    layer_type = "image"

    # load all files into array
    layer_data = []
    for _path in paths:
        # Read mrcfile as a memory mapped file
        data = mrcfile.mmap(_path, permissive=True).data

        # Append two layers if the data type is complex
        if data.dtype in [np.complex64, np.complex128]:
            layer_data.append((np.abs(data), {"name": "amplitude"}, layer_type))
            layer_data.append((np.angle(data), {"name": "phase"}, layer_type))
        else:
            layer_data.append((data, add_kwargs, layer_type))

    return layer_data
