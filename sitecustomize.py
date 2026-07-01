"""Course runtime defaults.

Python automatically imports this file when commands are run from the
course root. It keeps classroom demos portable without hardcoding any
machine-specific paths.
"""

import os


# Some macOS + Anaconda environments load two OpenMP runtimes when common
# ML packages are imported together. The demos are small classroom examples,
# so allowing duplicate OpenMP loading is the most practical cross-machine
# setting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")