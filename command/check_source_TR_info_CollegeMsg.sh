#!/bin/bash
set -e

sources=(259 319 673 86 274 711 272 554 617 439)

for source in "${sources[@]}"; do
    python -m app.check_source_TR_info --dataset_name CollegeMsg --source "$source"
done