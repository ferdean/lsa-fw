# This minimalistic image provides a fully functional, CPU-only, installation of
# DOLFINx (with PETSc) and common Python tooling

FROM dolfinx/dolfinx:stable

# @FIXME: The way to chose between real/complex builds seems to be something similar
# to what is commented here. It does change the scalar type (PETSc.ScalarType), but
# somehow breaks the unit tests and makes PETSc fail upon assembly of the operators,
# even in the real case:
# ```RuntimeError: Failed to successfully call PETSc function 'MatXIJSetPreallocation'.
# PETSc error code is: 63, Argument out of range```
# This error is found by trying to assemble any of the matrices defined in FEM.operators;
# both in the real and the complex cases.
# In case it is also a hint for debugging, importing a mesh file from a XDMF also
# crashes with this configuration
# (refer to `test/Meshing/test_core.py/test_facet_tags_export_import`).
# This is to be further investigated once complex results are needed (i.e., for the
# validation of the eigensolver)

# Select scalar mode (real or complex)
ARG FENICS_VARIANT=complex
ENV FENICS_VARIANT=${FENICS_VARIANT}
ENV DOLFINX_BASE=/usr/local/dolfinx-${FENICS_VARIANT}

# Define only for complex mode
ENV PETSC_ARCH=linux-gnu-complex128-32

# # Define only for real mode
# ENV PETSC_ARCH=linux-gnu-real64-32

# Environment variables for DOLFINx
ENV PKG_CONFIG_PATH=${DOLFINX_BASE}/lib/pkgconfig:$PKG_CONFIG_PATH
ENV PYTHONPATH=${DOLFINX_BASE}/lib/python3.12/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=${DOLFINX_BASE}/lib:$LD_LIBRARY_PATH

# Silence ZINK + fallback warnings globally
ENV LIBGL_ALWAYS_SOFTWARE=true \
    LIBGL_KOPPER_DISABLE=true \
    MESA_DEBUG=silent \
    MESA_ERROR_BACKTRACE_DISABLE=1 \
    MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

# Install extra Python packages
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    meshio \
    pyvista \
    jupyterlab \
    flake8 \
    black \
    isort \
    mypy \
    scipy \
    pytest \
    pandas

# Note for future devs:
# This build omits CUDA/GPU support because the current host lacks an NVIDIA GPU.
# To enable GPU acceleration in future once you have a CUDA-capable machine:
#   1. Switch the base image to an NVIDIA CUDA runtime, e.g.:
#        FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
#   2. Install any extra system dependencies
#   3. When building PETSc, add:
#        --with-cuda=1 \
#        --with-cuda-arch=<your_compute_capability> \
#        --download-kokkos --download-kokkos-kernels
#   4. Install and configure the NVIDIA Container Toolkit on the host.
#   5. Run your container with GPU access:
#        docker run --gpus all -it <your-image>
# Remember to consider real/complex scalar types when building PETSc manually.