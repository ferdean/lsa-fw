#!/usr/bin/env bash
set -e

if [[ "$1" == "complex" ]]; then
    cp .devcontainer/devcontainer-complex.json .devcontainer/devcontainer.json
    echo "Switched to COMPLEX devcontainer."
else
    cp .devcontainer/devcontainer-real.json .devcontainer/devcontainer.json
    echo "Switched to REAL devcontainer."
fi
echo "Environment change was successful!"
echo "    [!] Now run 'Dev Containers: Rebuild Container' from the VSCode command-palette."
echo ""
