"""Operations on Quantized (Integer) Voxel Arrays"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qvox")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
