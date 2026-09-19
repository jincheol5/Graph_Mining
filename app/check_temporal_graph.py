import argparse
from utils import DataUtils,GraphUtils
from graph import TemporalGraph

def app(**kwargs):
    """
    APP. Check temporal graph information.
    """
    graph_df=DataUtils.preprocess_temporal_graph_dataset(
        dataset_name=kwargs["dataset_name"]
    )
    nx_graph=GraphUtils.convert_temporal_graph_df_to_nx_graph(graph_df=graph_df)
    graph=TemporalGraph(
        graph_df=graph_df,
        graph=nx_graph,
        bipartite=kwargs["bipartite"]
    )
    print(f"Preprocessed Dataset Name: {kwargs['dataset_name']}")
    print(f"Is Bipartite?: {kwargs['bipartite']}")
    print(f"Number of Node: {graph.get_num_node()}")
    print(f"Number of Edge Events: {graph.get_num_edge_event()}")
    print(f"Number of Static Edge: {graph.get_num_static_edge()}")
    print(f"Min Timestamp: {graph.get_timestamp_info(info_type='min')}")
    print(f"Median Timestamp: {graph.get_timestamp_info(info_type='median')}")
    print(f"Max Timestamp: {graph.get_timestamp_info(info_type='max')}")
    print(f"Inductive node in test eventstream: {graph.check_inductivity()}")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
        type=str,
        choices=["CollegeMsg","bitcoin-alpha","bitcoin-otc","enron"],
        default=f"CollegeMsg"
    )
    parser.add_argument("--bipartite",type=int,default=0)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "bipartite":args.bipartite
    }
    app(**app_config)