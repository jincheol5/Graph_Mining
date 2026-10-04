#!/bin/bash
set -e

python -m app.check_temporal_graph_info --dataset_name CollegeMsg
python -m app.check_temporal_graph_info --dataset_name bitcoin-alpha
python -m app.check_temporal_graph_info --dataset_name bitcoin-otc