import pandas as pd
import networkx as nx
from typing import Literal

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
        self.median_t=graph_df["t"].median()

    def get_num_node(self)->int:
        return self.n_node

    def get_num_edge_event(self)->int:
        return self.n_event

    def get_num_static_edge(self)->int:
        return self.n_static_edge

    def get_timestamp_info(self,
            info_type:Literal["min","median","max"]
        ):
        match info_type:
            case "min":
                return self.min_t
            case "median":
                return self.median_t
            case "max":
                return self.max_t

    def check_inductivity(self,
            train_ratio:float=0.7
        )->int:
        """
        시간순으로 정렬한 이벤트 중 앞의 int(전체 행 수 * train_ratio)개를
        학습 구간으로 사용하고, 나머지 구간에서 처음 등장하는 고유 노드 수를
        반환한다. u와 i를 모두 포함하며, 같은 시각의 이벤트는 원래 순서를 유지한다.
        train_ratio는 0 이상 1 이하이다.
        """
        sorted_df=self.graph_df.sort_values("t",kind="stable")
        split_idx=int(len(sorted_df)*train_ratio)
        past_df=sorted_df.iloc[:split_idx]
        future_df=sorted_df.iloc[split_idx:]
        past_nodes=set(past_df["u"])|set(past_df["i"])
        future_nodes=set(future_df["u"])|set(future_df["i"])
        return len(future_nodes-past_nodes)
