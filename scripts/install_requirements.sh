#!/bin/bash

FILE=../requirements_ubuntu.txt

while read -r pkg; do
    [[ -z "$pkg" || "$pkg" =~ ^# ]] && continue
    echo "Installing $pkg..."
    pip install "$pkg" || echo "Failed to install: $pkg"
done < "$FILE"
