import pandas as pd
import networkx as nx
from typing import Literal

class TemporalGraphUtils:
    @staticmethod
    def convert_to_nx_graph(
            graph_df:pd.DataFrame,
            graph_type:Literal[
                "directed",
                "undirected"
            ]
        ):
        """
        Input:
            graph_df: pd.DataFrame, [u,i,t,idx]
        Return:
            graph: nx.MultiGraph or nx.MultiDiGraph, key=timestamp, attr=edge_id 
        """
        if graph_type=="directed":
            graph=nx.MultiDiGraph()
        if graph_type=="undirected":
            graph=nx.MultiGraph()
        for row in graph_df.itertuples(index=False):
            graph.add_edge(
                row.u,
                row.i,
                key=row.t,
                edge_id=row.idx,
            )
        return graph