#!/bin/bash
set -e

sources=(89 273 212 198 42 57 107 147 344 45)

for source in "${sources[@]}"; do
    python -m app.check_source_TR_info --dataset_name bitcoin-alpha --source "$source"
done