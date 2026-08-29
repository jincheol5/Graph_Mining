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
    nx_graph=GraphUtils.convert_temporal_graph_df_to_nx_graph(
        graph_df=graph_df,
        directed=kwargs["directed"]
    )
    graph=TemporalGraph(
        graph_df=graph_df,
        graph=nx_graph,
        directed=kwargs["directed"],
        bipartite=kwargs["bipartite"]
    )
    print(f"Dataset Name: {kwargs['dataset_name']}")
    print(f"Is Directed?: {kwargs['directed']}")
    print(f"Is Bipartite?: {kwargs['bipartite']}")
    print(f"Number of Node: {graph.get_num_node()}")
    print(f"Number of Edge Events: {graph.get_num_edge_event()}")
    print(f"Number of Static Edge: {graph.get_num_static_edge()}")
    timestamp_result=graph.get_min_max_timestamp()
    print(f"Min Timestamp: {timestamp_result['min_t']}")
    print(f"Max Timestamp: {timestamp_result['max_t']}")

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
    parser.add_argument("--directed",type=bool,default=True)
    parser.add_argument("--bipartite",type=bool,default=False)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "directed":args.directed,
        "bipartite":args.bipartite
    }
    app(**app_config)