from enum import Enum
from functools import partial

from napari import Viewer

from napari.components import ViewerModel
from copy import deepcopy

import numpy as np

from packaging.version import parse as parse_version

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Points, Layer
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info


NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


class SearchType(Enum):
    Highlight = 0
    Zoom_in = 1


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    """
    Create a copy of the layer and sets the viewer name. The data of the copied layer is shared with the original layer.

    Args:
        layer (Union[Labels, Image]): The layer to copy.
        name (Optional, str): The viewer name to set for the copied layer.

    Returns:
        Union[Labels, Image]: A copy of the layer.

    Notes:
        - If the layer is an Image or Labels layer, the data is not deeply copied.
        - This function assumes that the layer has a `metadata` attribute and an `events` attribute.
        - This function is optimized for napari version 0.4.16 or earlier.
    """
    # Shallow copy of the layer object
    res_layer = deepcopy(layer)

    # If the layer is an Image or Labels layer, share the data with the original layer
    if isinstance(layer, (Image, Points)):
        res_layer.data = layer.data

    # Set the viewer name for the copied layer
    res_layer.metadata["viewer_name"] = name

    # Disconnect the original layer's events and connect the copied layer's events
    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer

    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    # If napari version < 0.4.16, create a new layer object with the same data as the original layer
    res_layer = Layer.create(*layer.as_layer_data_tuple())

    # Share the data with the original layer if it's an Image or Labels layer
    if isinstance(layer, (Image, Points)):
        res_layer.data = layer.data

    # Set the viewer name for the copied layer
    res_layer.metadata["viewer_name"] = name

    return res_layer


def get_property_names(layer: Layer):
    class_ = layer.__class__
    res = []

    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue

        # Check if the event_name attribute is a property with a setter method
        if (
            isinstance(getattr(class_, event_name, None), property)
            and getattr(class_, event_name).fset is not None
        ):
            res.append(event_name)

    return res


def center_cross_on_mouse(
    viewer_model: napari.components.viewer_model.ViewerModel,
):
    """move the cross to the mouse position"""

    if not getattr(viewer_model, "mouse_over_canvas", True):
        # Notify the user that the mouse is not over the viewer canvas
        show_info("Mouse is not over the canvas. You may need to click on the canvas.")
        return

    viewer_model.dims.current_step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                    viewer_model.cursor.position, viewer_model.dims.range
                )
            ]
        ).astype(int)
    )


action_manager.register_action(
    name="napari:move_point",
    command=center_cross_on_mouse,
    description="Move dims point to mouse position",
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut("napari:move_point", "M")
action_manager.bind_shortcut("napari:add_point", "W")
action_manager.bind_shortcut("napari:add_positive", "A")
action_manager.bind_shortcut("napari:add_negative", "D")
action_manager.bind_shortcut("napari:remove", "S")


class OwnPartial:
    """
    A workaround for deepcopy not copying functools.partial objects.

    This class wraps a partial function, providing a __deepcopy__ method
    that returns a new instance of the wrapped function with the same
    arguments and keyword arguments.

    Note that this class is intended to be used as a last resort and should
    only be used when it is not possible to serialize the original object
    directly.
    """

    def __init__(self, func, *args, **kwargs):
        self.func = partial(func, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __deepcopy__(self, memodict={}):
        return OwnPartial(
            self.func.func,
            *deepcopy(self.func.args, memodict),
            **deepcopy(self.func.keywords, memodict),
        )


class QtViewerWrap(QtViewer):
    """
    A wrapper around the QtViewer class that provides drag-and-drop file
    opening functionality.

    This class stores a reference to a main viewer object and overrides the
    _qt_open method of the QtViewer class to open files in the main viewer.

    Args:
         main_viewer (Viewer): he main viewer object to which files should be opened.
        *args, **kwargs (Any): Additional arguments and keyword arguments to pass to the QtViewer
            constructor.
    """

    def __init__(self, main_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """
        Open the specified files in the main viewer object.

        Args:
            filenames (list[str]): The list of filenames to open.
            stack (bool): Whether to stack the images as layers.
            plugin (optional, str): The name of the plugin to use for opening the files (default is None).
            layer_type (optional, str): The layer type to use for opening the files (default is None).
            **kwargs (any): Additional keyword arguments to pass to the QtViewer _qt_open method.
        """
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )
