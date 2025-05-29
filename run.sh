#!/usr/bin/env bash

last_sif=$(ls -t apptainer/*.sif 2>/dev/null | head -n 1)

apptainer exec --writable-tmpfs $last_sif $@
