FROM dolfinx/dolfinx:stable

# Install extra Python packages
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    pyvista \
    jupyterlab \
    flake8 \
    black \
    isort \
    mypy \
    scipy \
    pandas
