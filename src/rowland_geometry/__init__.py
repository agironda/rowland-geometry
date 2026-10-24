# read version from installed package
from importlib.metadata import version
__version__ = version("rowland_geometry")

from .analyzer import Analyzer
from .frames import ReferenceFrame
from .rowland_circle import RowlandCircle
from .dispersive_refocusing import DispersiveRefocusing
from .rowland_spectrometer import RowlandSpectrometer
from .dispersive_spectrometer import DispersiveSpectrometer

__all__ = [
    "Analyzer",
    "ReferenceFrame",
    "RowlandCircle",
    "RowlandSpectrometer",
    "DispersiveRefocusing",
    "DispersiveSpectrometer"
]