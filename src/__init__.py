"""membrane protein localization for cryo-ET."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("membrain-pick")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lorenz Lamm"
__email__ = "lorenz.lamm@helmholtz-munich.de"
