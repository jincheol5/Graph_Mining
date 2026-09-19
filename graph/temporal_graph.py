import pandas as pd
import numpy as np
import networkx as nx
from typing import Literal

class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            nx_graph:nx.MultiDiGraph,
            bipartite:bool=False
        ):
        self.graph_df=graph_df # cols: [u, i, t, idx]
        self.nx_graph=nx_graph
        self.nx_static_graph=nx.DiGraph(nx_graph)
        self.bipartite=bipartite
        self.n_node=nx_graph.number_of_nodes()
        self.n_edge_event=nx_graph.number_of_edges()
        self.n_static_edge=self.nx_static_graph.number_of_edges()
        self.max_t=graph_df["t"].max()
        self.min_t=graph_df["t"].min()
        self.median_t=graph_df["t"].median()

        ### out_adj for algorithm
        out_adj=[[] for _ in range(self.n_node+1)]
        out_adj_edge=[[] for _ in range(self.n_node+1)]
        out_adj_t=[[] for _ in range(self.n_node+1)]
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

    def compute_SR(self,
            source:int,
            query_time:float|None=None,
            max_hop:int|None=None
        ):
        """
        Compute Static Reachability 
        - min_hop
        - Static Reachability라 하더라도 계산에 사용되는 edge 정보는 query time 이전 edge들로 한다.
        """
        if max_hop is None:
            max_hop=self.n_node-1

        INF=float("inf")
        reach_info={
            node:{
                "r":0,
                "hop":INF
            }
            for node in range(1,self.n_node+1)
        }
        reach_info[source]={
            "r":1,
            "hop":0
        }
        current={source}
        for hop in range(1,max_hop+1):
            next_state=set()
            for node in current:
                times=self.out_adj_t[node]
                neighbors=self.out_adj[node]
                end=(
                    len(times)
                    if query_time is None
                    else np.searchsorted(
                        times,
                        query_time,
                        side="right"
                    )
                )
                for idx in range(end):
                    dst=int(neighbors[idx])

                    # 이전 hop에서 이미 방문한 node
                    if reach_info[dst]["r"]==1:
                        continue
                    next_state.add(dst)

            if not next_state:
                break

            for node in next_state:
                reach_info[node]={
                    "r":1,
                    "hop":hop
                }
            current=next_state
        return reach_info

    def compute_TR(self,
            source:int,
            query_time:float|None=None,
            max_hop:int|None=None
        ):
        """
        Compute Temporal Reachability
        - Min hop
        """
        if max_hop is None:
            max_hop=self.n_node-1

        INF=float("inf")
        NEG_INF=float("-inf")

        TR_info={
            node:{
                "r":0,
                "hop":INF,
                "first_t":INF,
                "last_t":INF
            }
            for node in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":NEG_INF
        }

        # 탐색용 상태: node -> (arrival_t, first_t)
        current={
            source:(NEG_INF,0.0)
        }

        # 지금까지 각 node에 가장 일찍 도착한 시간
        best_arrival={
            source:NEG_INF
        }

        for hop in range(1,max_hop+1):
            next_state={}

            for node,(arrival_t,first_t) in current.items():
                times=self.out_adj_t[node]
                neighbors=self.out_adj[node]

                start=np.searchsorted(
                    times,
                    arrival_t,
                    side="right"
                )
                end=(
                    len(times)
                    if query_time is None
                    else np.searchsorted(
                        times,
                        query_time,
                        side="right"
                    )
                )

                for idx in range(start,end):
                    dst=int(neighbors[idx])
                    t=float(times[idx])

                    # 이전 hop에서 이미 더 일찍 도착했으면
                    # 현재 경로는 탐색 가치 없음
                    if best_arrival.get(dst,INF)<=t:
                        continue

                    # 같은 hop에서 이미 더 일찍 도착했으면 제거
                    if dst in next_state and next_state[dst][0]<=t:
                        continue

                    next_state[dst]=(
                        t,
                        t if hop==1 else first_t
                    )

            if not next_state:
                break

            for node,(arrival_t,first_t) in next_state.items():
                # 탐색용 earliest arrival은 계속 갱신
                best_arrival[node]=arrival_t

                # 결과는 최초 발견 때만 저장
                # hop-layer BFS이므로 최초 발견 hop = minimum hop
                if TR_info[node]["r"]==0:
                    TR_info[node]={
                        "r":1,
                        "hop":hop,
                        "first_t":first_t,
                        "last_t":arrival_t
                    }
            current=next_state
        return TR_info

    def get_graph_info(self)->dict:
        return {
            "n_node":self.n_node,
            "n_edge_event":self.n_edge_event,
            "n_static_edge":self.n_static_edge
        }

    def get_timestamp_info(self)->dict:
        return {
            "min_t":self.min_t,
            "median_t":self.median_t,
            "max_t":self.max_t
        }

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
