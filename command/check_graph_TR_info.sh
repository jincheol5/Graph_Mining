#!/bin/bash
set -e

python -m app.check_graph_TR_info --dataset_name CollegeMsg
python -m app.check_graph_TR_info --dataset_name bitcoin-alpha
python -m app.check_graph_TR_info --dataset_name bitcoin-otc
