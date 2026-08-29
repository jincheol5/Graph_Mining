import pandas as pd
import networkx as nx
from typing import Literal

class GraphUtils:
    @staticmethod
    def convert_temporal_graph_df_to_nx_graph(
            graph_df:pd.DataFrame,
            directed:bool=False
        ):
        """
        Input:
            graph_df: pd.DataFrame, [u,i,t,idx]
        Return:
            graph: nx.MultiGraph or nx.MultiDiGraph, key=timestamp, attr=edge_id 
        """
        if directed:
            graph=nx.MultiDiGraph()
        else:
            graph=nx.MultiGraph()
        for row in graph_df.itertuples(index=False):
            graph.add_edge(
                row.u,
                row.i,
                key=row.t,
                edge_id=row.idx,
            )
        return graph