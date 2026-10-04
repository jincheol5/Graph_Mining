#!/bin/bash
set -e

sources=(323 566 402 717 1402 1216 1212 440 1928 299)

for source in "${sources[@]}"; do
    python -m app.check_source_TR_info --dataset_name bitcoin-alpha --source "$source"
done