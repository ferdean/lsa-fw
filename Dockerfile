# This minimalistic image provides a fully functional, CPU-only, installation of
# DOLFINx (with PETSc) and common Python tooling

FROM dolfinx/dolfinx:stable

# Select scalar mode (real or complex)
ARG FENICS_VARIANT=complex
ENV FENICS_VARIANT=${FENICS_VARIANT}
ENV DOLFINX_BASE=/usr/local/dolfinx-${FENICS_VARIANT}

# Define only for complex mode
ENV PETSC_ARCH=linux-gnu-complex128-64

# Define only for real mode
# ENV PETSC_ARCH=linux-gnu-real64-64

# Environment variables for DOLFINx
ENV PKG_CONFIG_PATH=${DOLFINX_BASE}/lib/pkgconfig:$PKG_CONFIG_PATH
ENV PYTHONPATH=${DOLFINX_BASE}/lib/python3.12/dist-packages:$PYTHONPATH
ENV LD_LIBRARY_PATH=${DOLFINX_BASE}/lib:$LD_LIBRARY_PATH

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