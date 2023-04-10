from typing import Union, Sequence, List, Callable, Optional
from napari_ParticleAnnotation.utils.load_data import load_image, load_xyz
from napari.types import LayerData

PathOrPaths = Union[str, Sequence[str]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]


def get_reader_img(path: PathOrPaths) -> Optional[ReaderFunction]:
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    extensions = ".mrc", ".mrcs", ".map"
    if not path.endswith(extensions):
        return None

    # otherwise we return the *function* that can read ``path``.
    return load_image


def get_reader_xyz(path: PathOrPaths) -> Optional[ReaderFunction]:
    extensions = ".npy", ".csv", ".star"
    if not path.endswith(extensions):
        return None

    # otherwise we return the *function* that can read ``path``.
    return load_xyz
