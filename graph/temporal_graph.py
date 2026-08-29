import pandas as pd
import networkx as nx


class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            graph:nx.MultiGraph|nx.MultiDiGraph,
            directed:bool=False,
            bipartite:bool=False
        ):
        self.graph_df=graph_df # cols: [u, i, t, idx]
        self.graph=graph
        self.directed=directed
        self.bipartite=bipartite
        self.n_node=graph.number_of_nodes()
        self.n_event=graph.number_of_edges()
        if not directed:
            self.n_static_edge=nx.Graph(graph).number_of_edges()
        else:
            self.n_static_edge=nx.DiGraph(graph).number_of_edges()
        self.max_t=graph_df["t"].max()
        self.min_t=graph_df["t"].min()

    def get_num_node(self):
        return self.n_node

    def get_num_edge_event(self):
        return self.n_event

    def get_num_static_edge(self):
        return self.n_static_edge

    def get_min_max_timestamp(self):
        return {
            "min_t":self.min_t,
            "max_t":self.max_t
        }

