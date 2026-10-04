import pandas as pd
import numpy as np
import networkx as nx
import torch
from typing import Literal

class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        self.graph_df=graph_df # cols: [u, i, t, idx]
        self.bipartite=bipartite
        self.min_t=graph_df["t"].min()
        self.median_t=graph_df["t"].median()
        self.max_t=graph_df["t"].max()
        self.edge_events=[]

        ### convert to networkx graph
        self.graph=self.convert_df_to_nx_graph(graph_df=graph_df)
        self.n_node=self.graph.number_of_nodes()
        self.n_edge_event=self.graph.number_of_edges()
        self.static_graph=nx.DiGraph(self.graph)
        self.n_static_edge=self.graph.number_of_edges()

        ### out_adj for algorithm
        out_adj=[[] for _ in range(self.n_node+1)]
        out_adj_edge=[[] for _ in range(self.n_node+1)]
        out_adj_t=[[] for _ in range(self.n_node+1)]
        for event in graph_df.itertuples(index=False): # col: [u,i,t,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)

            # edge event 저장
            self.edge_events.append((src,dst,t,edge_id))

            # outgoing adj 저장
            out_adj[src].append(dst)
            out_adj_edge[src].append(edge_id)
            out_adj_t[src].append(t)

        ### convert list -> numpy array
        self.out_adj=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj
        ]
        self.out_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj_edge
        ]
        self.out_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in out_adj_t
        ]

    def convert_df_to_nx_graph(self,graph_df:pd.DataFrame):
        """
        Input:
            graph_df: pd.DataFrame, [u,i,t,idx]
        Return:
            graph: nx.MultiGraph or nx.MultiDiGraph, key=timestamp, attr=edge_id 
        """
        graph=nx.MultiDiGraph()
        for row in graph_df.itertuples(index=False):
            graph.add_edge(
                row.u,
                row.i,
                key=row.t,
                edge_id=row.idx,
            )
        return graph


