import argparse
from utils import DataUtils,GraphAnalysis
from graph import TemporalGraph

def app(**kwargs):
    """
    APP. Check temporal graph information.
    """
    graph_df=DataUtils.load_temporal_graph_df(dataset_name=kwargs["dataset_name"])
    graph=TemporalGraph(graph_df=graph_df)
    print(f"Preprocessed Dataset Name: {kwargs['dataset_name']}")
    print(f"Is Bipartite?: {graph.bipartite}")
    print(f"Number of Node: {graph.n_node}")
    print(f"Number of Edge Events: {graph.n_edge_event}")
    print(f"Number of Static Edge: {graph.n_static_edge}")
    print(f"Min Timestamp: {graph.min_t}")
    print(f"Median Timestamp: {graph.median_t}")
    print(f"Max Timestamp: {graph.max_t}")
    print(f"Inductive node in test eventstream: {GraphAnalysis.check_inductivity(graph_df=graph.graph_df,train_ratio=0.7)}")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "bitcoin-alpha",
            "bitcoin-otc",
            "enron"
        ],
        default=f"CollegeMsg"
    )
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name
    }
    app(**app_config)