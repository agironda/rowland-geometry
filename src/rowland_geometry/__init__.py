# read version from installed package
from importlib.metadata import version
__version__ = version("rowland_geometry")

from .analyzer import Analyzer
from .frames import ReferenceFrame
from .rowland_circle import RowlandCircle
from .spectrometer import RowlandSpectrometer

__all__ = [
    "Analyzer",
    "ReferenceFrame",
    "RowlandCircle",
    "RowlandSpectrometer",
]