import struct
from collections import namedtuple
import os

import numpy as np
import tifffile.tifffile as tiff
import torch
import starfile
import torch.nn.functional as F
import re
from ParticleAnnotation.utils.model.utils import get_device
from qtpy.QtWidgets import QFileDialog


def downsample(img: np.ndarray, factor=8):
    """Downsample 2d/3d array using fourier transform"""
    if factor == 1:
        return img

    if img.ndim == 3:
        img = F.interpolate(
            torch.Tensor(img.copy()).unsqueeze(0).unsqueeze(0),
            scale_factor=factor,
            mode="trilinear",
        ).numpy()[0, 0, ...]
    else:
        y, x = img.shape
        y = int(y / (1 / factor))
        x = int(x / (1 / factor))
        shape = (y, x)

        fft = np.fft.rfft2(img)

        A = fft[..., 0 : y // 2, 0 : x // 2 + 1]
        B = fft[..., -y // 2 :, 0 : x // 2 + 1]
        fft = np.concatenate([A, B], axis=0)

        # scale the signal from downsampling
        a = x * y
        b = img.shape[-2] * img.shape[-1]
        fft *= a / b

        return np.fft.irfft2(fft, s=shape).astype(img.dtype)
    return img.astype(img.dtype)


def load_coordinates(path):
    """
    Load coordinates from a file.
    """
    if path.endswith(".csv"):
        data = np.genfromtxt(path, delimiter=",", dtype=float)

    elif path.endswith(".star"):
        # Implement reading of .star file
        data = starfile.read(path)
        x = data["rlnCoordinateX"]
        y = data["rlnCoordinateY"]
        z = data["rlnCoordinateZ"]
        data = np.column_stack((x, y, z))
        return None, None

    elif path.endswith(".txt"):
        data = np.genfromtxt(path, delimiter=",", dtype=float)

    elif path.endswith(".npy"):
        data = np.load(path)

    elif path.endswith(".tbl"):
        # Implement reading of .tbl file
        data = np.genfromtxt(path, delimiter=",", dtype=float)
        # this will have nans where there are strings
        return None, None

    else:
        print("Could not load coordinates from file")
        return None, None

    data = data[~np.isnan(data).any(axis=1)]
    # if data is x,y,z just return
    if data.shape[1] == 3:
        labels = np.ones(data.shape[0])

    if data.shape[1] == 4:
        # labels would either be the first or the last column
        if np.all(data[:, 3] == data[:, 3].astype(int)):
            labels = data[:, 3].astype(int)
            data = data[:, 0:3]
            
        else:
            if np.all(data[:, 0] == data[:, 0].astype(int)):
                data = data[:, 1:4]
                labels = data[:, 0].astype(int)
            else:
                labels = np.ones(data.shape[0])
                data = data[:, 0:3]

    # if data is index,x,y,z,label - this is how napari saves!
    if data.shape[1] == 5:
        labels = data[:, 4]
        # check if the labels are integers
        if np.all(labels == labels.astype(int)):
            labels = labels.astype(int)
        else:
            labels = np.ones(data.shape[0])
        data = data[:, 1:4]

    data = np.array(
            (
                np.array(labels).astype(np.int16),
                data[:, 0],
                data[:, 1],
                data[:, 2],
            )
        ).T

    return data, labels


def save_coordinates(path, data):
    """
    Save coordinates to a file.
    """
    # add header
    header = ["axis_0", "axis_1", "axis_2"]
    data = np.vstack([header, data])
    np.savetxt(path, data, delimiter=",", fmt="%s")


def load_template():
    """
    Load the template scores from disk.

    Args:
        path: A string representing the path of the file to read.

    Returns:
        The template score data.

    """
    device_ = get_device()
    # temp_name = temp_name.upper()
    # tomo_name = re.search(r"ts(\d{1,3})", path).group(0)

    # [TO-DO] Remove down sampling after testing
    # root = f'/h2/njain/data/tomonet_template_matched/downsampled'
    root = QFileDialog.getOpenFileName(None, "Select a score file")[0]
    print(root)

    # root = f"/Users/navyajain/napari-ParticleAnnotation-1/test_images/"
    # print(f"Found template name as - {tomo_name}")
    template_score = torch.load(root, map_location=device_
                                )
    print("Loaded template scores")
    # try:
    #     template_score = torch.load(root, map_location=device_
    #     ).numpy()
    #     # flip the template score along the y-axis
    #     # ice_score = torch.load(
    #     #     f"{root}/scores_ice.pt", map_location=device_
    #     # ).numpy()
    #
    #     # We want to load just one file which is already ready
    #     # template_score = np.concatenate([template_score, ice_score], axis=0)
    #     # template_score = np.flip(template_score, axis=2)
    #     print("Loaded template scores")
    # except:
    #     print(f"Could not find template {root}")
    #     # template_score = torch.load(
    #     #     root, map_location=device_
    #     # ).numpy()
    #     # ice_score = torch.load(
    #     #     f"{root}/scores_ice.pt", map_location=device_
    #     # ).numpy()
    #     # template_score = np.concatenate([template_score, ice_score], axis=0)
    #     # template_score = np.flip(template_score, axis=2)

    return template_score


def load_image(path, aws=False):
    """
    Load an image file from disk.

    Args:
        path: A string or sequence of strings representing the path(s) of the file(s) to read.
        aws: Load data from AWS

    Returns:
        A list of tuples, where each tuple contains the image data, additional keyword arguments, and the layer type.
        If the image data is complex, two tuples will be returned for the amplitude and phase components.
    """
    paths = [path] if isinstance(path, str) else path
    layer_data = []
    for _path in paths:
        # Read mrcfile as a memory mapped file
        if _path.endswith((".mrc", ".rec")):
            data, px = load_mrc_file(_path)
        elif _path.endswith(".tif"):
            data = tiff.imread(_path)
            px = 1.0
        elif _path.endswith(".am"):
            data = import_am(_path)
            px = 1.0

        if aws:
            return data

        # Append two layers if the data type is complex
        if np.issubdtype(data.dtype, np.complexfloating):
            layer_data.append((np.abs(data), {"name": "amplitude"}, "image"))
            layer_data.append((np.angle(data), {"name": "phase"}, "image"))
        else:
            # data = data[0:250, data.shape[1]//2-128:data.shape[1]//2+128, data.shape[2]//2-128:data.shape[2]//2+128]
            layer_data.append((data, {}, "image"))

    print(f"Loaded {_path} with {px} pixel size")
    return layer_data


def load_data_aws(image):
    layer_data = []

    # Append two layers if the data type is complex
    if np.issubdtype(image.dtype, np.complexfloating):
        layer_data.append((np.abs(image), {"name": "amplitude"}, "image"))
        layer_data.append((np.angle(image), {"name": "phase"}, "image"))
    else:
        layer_data.append((image, {}, "image"))

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
    labels = None

    for _path in paths:
        # Load Numpy
        if _path.endswith(".npy"):
            xyz = np.load(_path)

        # Load CSV
        if _path.endswith(".csv"):
            xyz = np.genfromtxt(_path, delimiter=",", dtype=float)
            if xyz.shape[1] == 4:
                labels = xyz[:, 0]
                xyz = xyz[:, 1:]

        # Load Star
        if _path.endswith(".star"):
            xyz = None  # TODO: implement support for Star format

        if isinstance(xyz[0, 0], str):
            xyz = xyz[1:, :]

        if labels is None:
            layer_data.append((xyz, {}, "points"))
        else:
            layer_data.append((xyz, {"label": labels}, "points"))
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


# int nx
# int ny
# int nz
fstr = "3i"
names = "nx ny nz"

# int mode
fstr += "i"
names += " mode"

# int nxstart
# int nystart
# int nzstart
fstr += "3i"
names += " nxstart nystart nzstart"

# int mx
# int my
# int mz
fstr += "3i"
names += " mx my mz"

# float xlen
# float ylen
# float zlen
fstr += "3f"
names += " xlen ylen zlen"

# float alpha
# float beta
# float gamma
fstr += "3f"
names += " alpha beta gamma"

# int mapc
# int mapr
# int maps
fstr += "3i"
names += " mapc mapr maps"

# float amin
# float amax
# float amean
fstr += "3f"
names += " amin amax amean"

# int ispg
# int next
# short creatid
fstr += "2ih"
names += " ispg next creatid"

# pad 30 (extra data)
# [98:128]
fstr += "30x"

# short nint
# short nreal
fstr += "2h"
names += " nint nreal"

# pad 20 (extra data)
# [132:152]
fstr += "20x"

# int imodStamp
# int imodFlags
fstr += "2i"
names += " imodStamp imodFlags"

# short idtype
# short lens
# short nd1
# short nd2
# short vd1
# short vd2
fstr += "6h"
names += " idtype lens nd1 nd2 vd1 vd2"

# float[6] tiltangles
fstr += "6f"
names += " tilt_ox tilt_oy tilt_oz tilt_cx tilt_cy tilt_cz"

# NEW-STYLE MRC image2000 HEADER - IMOD 2.6.20 and above
# float xorg
# float yorg
# float zorg
# char[4] cmap
# char[4] stamp
# float rms
fstr += "3f4s4sf"
names += " xorg yorg zorg cmap stamp rms"

# int nlabl
# char[10][80] labels
fstr += "i800s"
names += " nlabl labels"

header_struct = struct.Struct(fstr)
MRCHeader = namedtuple("MRCHeader", names)


def mrc_read_header(mrc):
    """
    Helper function to read MRC header.

    Args:
        mrc (str): MRC file directory.

    Returns:
        class: MRC header.
    """
    if isinstance(mrc, str):
        with open(mrc, "rb") as f:
            header = f.read(1024)
    else:
        header = mrc

    return MRCHeader._make(header_struct.unpack(header))


def mrc_write_header(*args) -> bytes:
    header = MRCHeader(*args)
    return header_struct.pack(*list(header))


def mrc_mode(mode: int, amin: int):
    """
    Helper function to decode MRC mode type.

    mode int: MRC mode from mrc header.
    amin int: MRC minimum pixel value.

    Returns:
        np.dtype: Mode as np.dtype.
    """
    dtype_ = {
        0: np.uint8,
        1: np.int16,  # Signed 16-bit integer
        2: np.float32,  # Signed 32-bit real
        3: "2h",  # Complex 16-bit integers
        4: np.complex64,  # Complex 32-bit reals
        6: np.uint16,  # Unassigned int16
        12: np.float16,  # Signed 16-bit half-precision real
        16: "3B",  # RGB values
    }

    if isinstance(mode, int):
        if mode == 0 and amin >= 0:
            return dtype_[mode]
        elif mode == 0 and amin < 0:
            return np.int8

        if mode in dtype_:
            return dtype_[mode]
    else:
        if mode in [np.int8, np.uint8]:
            return 0
        for name in dtype_:
            if mode == dtype_[name]:
                return name


def load_mrc_file(mrc: str):
    """
    Function to load MRC 2014 file format.

    Args:
        mrc (str): MRC file directory.

    Returns:
        np.ndarray, float: Image data and pixel size.
    """
    header = mrc_read_header(mrc)
    extended_header = header.next

    pixel_size = round(header.xlen / header.nx, 3)
    dtype = mrc_mode(header.mode, header.amin)
    nz, ny, nx = header.nz, header.ny, header.nx
    bit_len = nz * ny * nx

    # Check for corrupted files
    try:
        if nz == 1:
            image = np.fromfile(mrc, dtype=dtype)[-bit_len:].reshape((ny, nx))
        else:
            image = np.fromfile(mrc, dtype=dtype)[-bit_len:].reshape((nz, ny, nx))
    except ValueError:  # File is corrupted
        if nz > 1:
            if mrc.endswith(".rec"):
                header_len = 512
            else:
                header_len = 1024 + extended_header
            image = np.fromfile(mrc, dtype=dtype)[header_len:]

            while bit_len >= len(image):
                nz = nz - 1
                bit_len = nz * ny * nx

            image = image[:bit_len]
            image = image.reshape((nz, ny, nx))
        else:
            image = None

    if image is None:
        return None, 1.0

    if image.min() < 0 and image.dtype == np.int8:
        image = image + 128
        image = image.astype(np.uint8)

    if image.min() < 0 and image.dtype == np.int16:
        image = image + 32768
        image = image.astype(np.uint16)

    if nz > ny:
        image = image.transpose((1, 0, 2))  # YZX to ZYX
    elif nz > nx:
        image = image.transpose((2, 1, 0))  # XYZ to ZYX

    image = np.flip(image, 1)
    return image, pixel_size
