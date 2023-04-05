from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget


class SelectData(QWidget):
    sig_update_hist = Signal(object)
