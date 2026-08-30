import os
import pandas as pd
import numpy as np
from typing import Literal

class DataUtils:
    base_path=os.path.join("..","data")
    @staticmethod
    def preprocess_SNAP_temporal_graph_dataset(
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-otc",
                "bitcoin-alpha"
            ]
        ):
        """
        Return:
            graph_df: pd.DataFrame
        """
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(DataUtils.base_path,"temporal_graph",dataset_name,f"raw_{dataset_name}.txt")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","t"],
                )
            case "bitcoin-otc"|"bitcoin-alpha":
                dataset_path=os.path.join(DataUtils.base_path,"temporal_graph",dataset_name,f"raw_{dataset_name}.csv")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,3],
                    names=["u","i","t"],
                )

        # 결측값 제거
        graph_df=graph_df.dropna(
            subset=["u","i","t"]
        ).copy()

        # 자료형 변환
        graph_df["u"]=graph_df["u"].astype(int)
        graph_df["i"]=graph_df["i"].astype(int)
        # SNAP timestamp(초)를 Unix timestamp의 일(day) 단위 정수로 변환
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .floordiv(24*60*60)
            .astype(np.int64)
        )

        # self-loop 제거
        self_loop_mask=graph_df["u"]==graph_df["i"]
        graph_df=graph_df.loc[~self_loop_mask].copy()

        # 동일한 시각의 동일한 방향 edge(u -> i)는 하나만 유지
        graph_df=(
            graph_df
            .drop_duplicates(subset=["u","i","t"],keep="first")
            .reset_index(drop=True)
        )

        # 첫 interaction 시각을 0으로 맞춤
        if not graph_df.empty:
            graph_df["t"]=graph_df["t"]-graph_df["t"].min()

        # 모든 node ID 수집
        node_ids=sorted(
            set(graph_df["u"]).union(graph_df["i"])
        )

        # 기존 node ID -> 1 ~ N 매핑
        node_mapping={
            original_id:mapped_id
            for mapped_id,original_id in enumerate(
                node_ids,
                start=1,
            )
        }

        # node ID 재매핑
        graph_df["u"]=graph_df["u"].map(node_mapping).astype(int)
        graph_df["i"]=graph_df["i"].map(node_mapping).astype(int)

        # edge ID를 1 ~ E로 지정
        graph_df["idx"]=range(1,len(graph_df)+1)
        return graph_df

    @staticmethod
    def preprocess_ZENODO_temporal_graph_dataset(
            dataset_name:Literal[
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        Return:
            graph_df: pd.DataFrame
        """
        graph_path=os.path.join(DataUtils.base_path,"temporal_graph",dataset_name,f"ml_{dataset_name}.csv")
        graph_df=(
            pd.read_csv(graph_path,index_col=0)[["u","i","ts","idx"]]
            .rename(columns={"ts": "t"})
            .astype({
                "u": int,
                "i": int,
                "idx": int,
            })
        )

        # timestamp(초)를 Unix timestamp의 일(day) 단위 정수로 변환
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .floordiv(24*60*60)
            .astype(np.int64)
        )

        ### remove self-loop, 동일한 시각의 동일한 방향 edge(u -> i)는 하나만 유지
        graph_df=(
            graph_df[graph_df["u"]!=graph_df["i"]]
            .drop_duplicates(subset=["u","i","t"],keep="first")
            .reset_index(drop=True)
        )

        # 첫 interaction 날짜를 0으로 맞춤
        if not graph_df.empty:
            graph_df["t"]=graph_df["t"]-graph_df["t"].min()

        ### remap edge index
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        # remap node id
        used_nodes=np.sort(np.unique(graph_df[["u","i"]].to_numpy()))
        node_map={
            old_id:new_id
            for new_id,old_id
            in enumerate(used_nodes,start=1)
        }
        graph_df[["u","i"]]=(
            graph_df[["u","i"]]
            .replace(node_map)
            .astype(int)
        )
        return graph_df

    @staticmethod
    def preprocess_temporal_graph_dataset(
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-otc",
                "bitcoin-alpha",
                "enron",
                "wikipedia",
                "reddit"
            ]
        ):
        """
        Return:
            graph_df: pd.DataFrame
        """
        match dataset_name:
            case "CollegeMsg"|"bitcoin-otc"|"bitcoin-alpha":
                return DataUtils.preprocess_SNAP_temporal_graph_dataset(dataset_name=dataset_name)
            case "enron"|"wikipedia"|"reddit":
                return DataUtils.preprocess_ZENODO_temporal_graph_dataset(dataset_name=dataset_name)
