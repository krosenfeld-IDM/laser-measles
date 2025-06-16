#!/bin/bash

# Find all .py files in the current directory and convert them to notebooks
for py_file in *.py; do
    if [ -f "$py_file" ]; then
        echo "Converting $py_file to notebook..."
        jupytext --to notebook "$py_file"
    fi
done 