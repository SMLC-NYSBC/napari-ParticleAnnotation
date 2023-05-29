import numpy as np
import mrcfile
import tifffile.tifffile as tiff


def downsample(img, factor=8):
    """Downsample 2d/3d array using fourier transform"""
    if factor == 1:
        return img

    if len(img.shape) == 3:
        z, y, x = img.shape
    else:
        y, x = img.shape
        z = 1
    y = int(y / factor)
    x = int(x / factor)
    shape = (y, x)

    if z > 1:
        f_img = np.zeros((z, y, x))
        for i in range(z):
            F = np.fft.rfft2(img[i, ...])

            A = F[..., 0: y // 2, 0: x // 2 + 1]
            B = F[..., -y // 2:, 0: x // 2 + 1]
            F = np.concatenate([A, B], axis=0)

            # scale the signal from downsampling
            a = x * y
            b = img[i, ...].shape[-2] * img[i, ...].shape[-1]
            F *= a / b

            f_img[i, ...] = np.fft.irfft2(F, s=shape)
        return f_img.astype(img.dtype)
    else:
        F = np.fft.rfft2(img)

        A = F[..., 0: y // 2, 0: x // 2 + 1]
        B = F[..., -y // 2:, 0: x // 2 + 1]
        F = np.concatenate([A, B], axis=0)

        # scale the signal from downsampling
        a = x * y
        b = img.shape[-2] * img.shape[-1]
        F *= a / b

        return np.fft.irfft2(F, s=shape).astype(img.dtype)


def load_image(path):
    """
    Load an image file from disk.

    Args:
        path: A string or sequence of strings representing the path(s) of the file(s) to read.

    Returns:
        A list of tuples, where each tuple contains the image data, additional keyword arguments, and the layer type.
        If the image data is complex, two tuples will be returned for the amplitude and phase components.
    """
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for _path in paths:
        # Read mrcfile as a memory mapped file
        if _path.endswith((".mrc", ".rec")):
            data = mrcfile.mmap(_path, permissive=True).data
        elif _path.endswith(".tif"):
            data = tiff.imread(_path)
        elif _path.endswith(".am"):
            data = import_am(_path)

        # Append two layers if the data type is complex
        if np.issubdtype(data.dtype, np.complexfloating):
            layer_data.append((np.abs(data), {"name": "amplitude"}, "image"))
            layer_data.append((np.angle(data), {"name": "phase"}, "image"))
        else:
            layer_data.append((data, {}, "image"))
    return layer_data


def load_xyz(path):
    """
    Load an XYZ point cloud file from disk.

    Args:
        path: A string or sequence of strings representing the path(s) of the file(s) to read.

    Returns:
        A list of tuples, where each tuple contains the point cloud data, additional keyword arguments, and the layer type.
    """
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    xyz = None

    for _path in paths:
        # Load Numpy
        if _path.endswith(".npy"):
            xyz = np.load(_path)

        # Load CSV
        if _path.endswith(".csv"):
            xyz = np.genfromtxt(_path, delimiter=",")

        # Load Star
        if _path.endswith(".star"):
            xyz = None  # TODO: implement support for Star format

        if isinstance(xyz[0, 0], str):
            xyz = xyz[1:, :]

        if xyz.shape[1] == 3:  # No labels
            xyz_layer = np.zeros((len(xyz), 5))
            xyz_layer[:, 2:] = xyz
        elif xyz.shape[1] == 4:  # Labels
            xyz_layer = np.zeros((len(xyz), 5))
            xyz_layer[:, 1:] = xyz

        layer_data.append((xyz_layer, {}, "points"))
    return layer_data


def import_am(am_file: str):
    """
    Function to load Amira binary image data.

    Args:
        am_file (str): Amira binary image .am file.

    Returns:
        np.ndarray, float, float, list: Image file as well images parameters.
    """
    am = open(am_file, "r", encoding="iso-8859-1").read(8000)

    asci = False
    if "AmiraMesh 3D ASCII" in am:
        assert "define Lattice" in am
        asci = True

    size = [word for word in am.split("\n") if word.startswith("define Lattice ")][0][
        15:
    ].split(" ")

    nx, ny, nz = int(size[0]), int(size[1]), int(size[2])

    if "Lattice { byte Data }" in am:
        if asci:
            img = (
                open("../../rand_sample/T216_grid3b.am", "r", encoding="iso-8859-1")
                .read()
                .split("\n")
            )
            img = [x for x in img if x != ""]
            img = np.asarray(img)
            return img
        else:
            img = np.fromfile(am_file, dtype=np.uint8)

    elif "Lattice { sbyte Data }" in am:
        img = np.fromfile(am_file, dtype=np.int8)
        img = img + 128

    binary_start = str.find(am, "\n@1\n") + 4
    img = img[binary_start:-1]
    if nz == 1:
        if len(img) == ny * nx:
            img = img.reshape((ny, nx))
        else:
            df_img = np.zeros((ny * nx), dtype=np.uint8)
            df_img[: len(img)] = img
            img = df_img.reshape((ny, nx))
    else:
        if len(img) == nz * ny * nx:
            img = img.reshape((nz, ny, nx))
        else:
            df_img = np.zeros((nz * ny * nx), dtype=np.uint8)
            df_img[: len(img)] = img
            img = df_img.reshape((nz * ny * nx))

    return img
