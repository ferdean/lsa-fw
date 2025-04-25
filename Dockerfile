# This minimalistic image provides a fully functional, CPU-only, installation of
# DOLFINx (with PETSc) and common Python tooling

FROM dolfinx/dolfinx:stable

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