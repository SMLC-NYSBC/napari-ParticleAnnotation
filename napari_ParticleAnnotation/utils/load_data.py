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


def load_xyz(path):
    layer_type = "points"
    add_kwargs = {}

    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for _path in paths:
        # Load Numpy
        if path.endswith(".npy"):
            xyz = np.load(path)

        # Load CSV
        if path.endswith(".csv"):
            xyz = np.genfromtxt(path, delimiter=",")

        # Load Star
        if path.endswith(".star"):
            xyz = None

        assert xyz.ndim == 2 and xyz.shape[1] in [2, 3], "Need 2D or 3D point cloud!"

        layer_data.append((xyz, add_kwargs, layer_type))
