import pandas as pd
import torch

class GraphAnalysis:
    @staticmethod
    def check_inductivity(
            graph_df:pd.DataFrame,
            train_ratio:float=0.7
        )->int:
        """
        시간순으로 정렬한 이벤트 중 앞의 int(전체 행 수 * train_ratio)개를 학습 구간으로 사용하고, 
        나머지 구간에서 처음 등장하는 고유 노드 수를 반환한다. 
        u와 i를 모두 포함하며, 같은 시각의 이벤트는 원래 순서를 유지한다.
        train_ratio는 0 이상 1 이하이다.
        """
        sorted_df=graph_df.sort_values("t",kind="stable",inplace=False)
        split_idx=int(len(sorted_df)*train_ratio)
        past_df=sorted_df.iloc[:split_idx]
        future_df=sorted_df.iloc[split_idx:]
        past_nodes=set(past_df["u"])|set(past_df["i"])
        future_nodes=set(future_df["u"])|set(future_df["i"])
        return len(future_nodes-past_nodes)

    @staticmethod
    def check_source_TR_ratio_info(
            source:int,
            TR_label:torch.Tensor
        )->list[float]:
        """
        batch sequence의 10%, 20%, ..., 100% 지점에서 TR ratio를 확인한다.
        각 지점은 ceil(batch_seq_len * 비율) - 1 인덱스를 사용한다.
        0번 패딩과 source 자신은 분자 및 분모에서 제외한다.
        시퀀스 길이가 10보다 작으면 일부 지점을 반복 사용한다.
        목적지 노드가 없으면 각 비율은 0.0이다.

        Input:
            source: int (1~N)
            TR_label: [batch_seq_len,N+1,N+1] bool tensor
        Return:
            TR_ratio_info: list[float], 길이 10, 각 값은 0~1
        """
        seq_len=TR_label.shape[0]
        n_node=TR_label.shape[2]-1
        TR_ratio_info=[]
        for part in range(1,11):
            time_idx=(seq_len*part+9)//10-1
            reachable=TR_label[time_idx,source,1:].clone()
            reachable[source-1]=False
            ratio=reachable.sum().item()/(n_node-1)
            TR_ratio_info.append(ratio)
        return TR_ratio_info

    @staticmethod
    def check_source_TR_hop_info(
            source:int,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        )->dict:
        """
        Input:
            source: int (1~N)
            TR_label: [batch_seq_len,N+1,N+1] bool tensor
            TR_hop: [batch_seq_len,N+1,N+1] int tensor
        Return:
            hop_info: dict
                min_hop
                n_min_hop
                max_hop
                n_max_hop
                mean_hop
        """
        last_TR_label=TR_label[-1,source,:]
        last_TR_hop=TR_hop[-1,source,:]
        reachable=last_TR_label.clone()
        reachable[0]=False
        reachable[source]=False
        hops=last_TR_hop[reachable]

        if hops.numel()==0:
            return {
                "min_hop":None,
                "n_min_hop":0,
                "max_hop":None,
                "n_max_hop":0,
                "mean_hop":None
            }

        min_hop=hops.min().item()
        max_hop=hops.max().item()
        return {
            "min_hop":min_hop,
            "n_min_hop":(hops==min_hop).sum().item(),
            "max_hop":max_hop,
            "n_max_hop":(hops==max_hop).sum().item(),
            "mean_hop":hops.to(dtype=torch.float64).mean().item()
        }

    @staticmethod
    def check_graph_TR_hop_info(
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        )->dict:
        """
        Input:
            TR_label: [batch_seq_len,N+1,N+1] bool tensor
            TR_hop: [batch_seq_len,N+1,N+1] int tensor
        Return:
            hop_info: dict
                min_hop
                max_hop
                mean_hop
        """
        last_TR_label=TR_label[-1,:,:]
        last_TR_hop=TR_hop[-1,:,:]
        reachable=last_TR_label.clone()
        reachable[0,:]=False
        reachable[:,0]=False
        reachable.fill_diagonal_(False)
        hops=last_TR_hop[reachable]

        if hops.numel()==0:
            return {
                "min_hop":None,
                "n_min_hop":0,
                "max_hop":None,
                "n_max_hop":0,
                "mean_hop":None
            }

        min_hop=hops.min().item()
        max_hop=hops.max().item()
        return {
            "min_hop":min_hop,
            "max_hop":max_hop,
            "mean_hop":hops.to(dtype=torch.float64).mean().item()
        }