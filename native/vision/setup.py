"""Build the omni_vision extension against the system OpenCV (pkg-config opencv4).

sudo apt-get install libopencv-dev       # Debian/Ubuntu
pip install ./native/vision
"""

import shlex
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def pkg_config(*args: str) -> list[str]:
    out = subprocess.run(
        ["pkg-config", *args, "opencv4"], check=True, capture_output=True, text=True
    )
    return shlex.split(out.stdout)


cflags = pkg_config("--cflags")
libs = pkg_config("--libs-only-l")
# Only the modules the extension uses.
wanted = ("-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs")
ext = Pybind11Extension(
    "omni_vision",
    ["src/omni_vision.cpp"],
    cxx_std=17,
    include_dirs=[f[2:] for f in cflags if f.startswith("-I")],
    library_dirs=[f[2:] for f in pkg_config("--libs-only-L") if f.startswith("-L")],
    # Only the modules used: core, imgproc, imgcodecs.
    libraries=[
        lib[2:]
        for lib in libs
        if lib in ("-lopencv_core", "-lopencv_imgproc", "-lopencv_imgcodecs")
    ],
    extra_compile_args=["-O3"],
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
