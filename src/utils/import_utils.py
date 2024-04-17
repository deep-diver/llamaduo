import sys 
import importlib 

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


_alignment_available = importlib.util.find_spec("alignment") is not None
try:
    _alignment_version = importlib_metadata.version("alignment")
except importlib_metadata.PackageNotFoundError:
    _alignment_available = False

def is_alignment_available():
    return _alignment_available

