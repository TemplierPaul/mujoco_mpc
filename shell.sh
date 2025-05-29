#!/usr/bin/env bash

last_sif=$(ls -t apptainer/*.sif 2>/dev/null | head -n 1)

# Shell into the container
apptainer shell --writable-tmpfs $last_sif
