import argparse
import torch
from utils import DataUtils,GraphAnalysis
from graph import TemporalGraph

def app(**kwargs):
    """
    APP. Check temporal reachability information.
    """
    ### get app parameter 
    dataset_name=kwargs["dataset_name"]
    source=kwargs["source"]

    ### load TR_result
    train_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="train",
        batch_size=200
    )
    val_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="val",
        batch_size=200
    )
    test_TR_result=DataUtils.load_TR_result(
        dataset_name=dataset_name,
        purpose="test",
        batch_size=200
    )

    ### get TR_label
    train_TR_label=train_TR_result["label"]
    val_TR_label=val_TR_result["label"]
    test_TR_label=test_TR_result["label"]
    TR_label=torch.cat(
        [
            train_TR_label,
            val_TR_label,
            test_TR_label
        ],
        dim=0
    )

    ### get TR_hop
    train_TR_hop=train_TR_result["hop"]
    val_TR_hop=val_TR_result["hop"]
    test_TR_hop=test_TR_result["hop"]
    TR_hop=torch.cat(
        [
            train_TR_hop,
            val_TR_hop,
            test_TR_hop
        ],
        dim=0
    )

    ### Analysis: TR ratio
    TR_ratio_info=GraphAnalysis.check_source_TR_ratio_info(source=source,TR_label=TR_label)
    print(f"Dataset {dataset_name} Source {source}의 Batch 시퀀스의 TR ratio 변화:")
    print(TR_ratio_info,end="\n\n")

    ### Analysis: TR hop info
    train_TR_hop_info=GraphAnalysis.check_source_TR_hop_info(source=source,TR_label=train_TR_label,TR_hop=train_TR_hop)
    print(f"Dataset {dataset_name} Source {source}의 Train eventstream의 TR hop 정보:")
    print(f"min_hop: {train_TR_hop_info['min_hop']}")
    print(f"n_min_hop: {train_TR_hop_info['n_min_hop']}")
    print(f"max_hop: {train_TR_hop_info['max_hop']}")
    print(f"n_max_hop: {train_TR_hop_info['n_max_hop']}")
    print(f"mean_hop: {train_TR_hop_info['mean_hop']}",end="\n\n")

    val_TR_hop_info=GraphAnalysis.check_source_TR_hop_info(source=source,TR_label=val_TR_label,TR_hop=val_TR_hop)
    print(f"Dataset {dataset_name} Source {source}의 Validate eventstream의 TR hop 정보:")
    print(f"min_hop: {val_TR_hop_info['min_hop']}")
    print(f"n_min_hop: {val_TR_hop_info['n_min_hop']}")
    print(f"max_hop: {val_TR_hop_info['max_hop']}")
    print(f"n_max_hop: {val_TR_hop_info['n_max_hop']}")
    print(f"mean_hop: {val_TR_hop_info['mean_hop']}",end="\n\n")

    test_TR_hop_info=GraphAnalysis.check_source_TR_hop_info(source=source,TR_label=test_TR_label,TR_hop=test_TR_hop)
    print(f"Dataset {dataset_name} Source {source}의 Test eventstream의 TR hop 정보:")
    print(f"min_hop: {test_TR_hop_info['min_hop']}")
    print(f"n_min_hop: {test_TR_hop_info['n_min_hop']}")
    print(f"max_hop: {test_TR_hop_info['max_hop']}")
    print(f"n_max_hop: {test_TR_hop_info['n_max_hop']}")
    print(f"mean_hop: {test_TR_hop_info['mean_hop']}",end="\n\n")

    TR_hop_info=GraphAnalysis.check_source_TR_hop_info(source=source,TR_label=TR_label,TR_hop=TR_hop)
    print(f"Dataset {dataset_name} Source {source}의 All eventstream의 TR hop 정보:")
    print(f"min_hop: {TR_hop_info['min_hop']}")
    print(f"n_min_hop: {TR_hop_info['n_min_hop']}")
    print(f"max_hop: {TR_hop_info['max_hop']}")
    print(f"n_max_hop: {TR_hop_info['n_max_hop']}")
    print(f"mean_hop: {TR_hop_info['mean_hop']}",end="\n\n")

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
    parser.add_argument("--source",type=int,default=1)
    args=parser.parse_args()
    app_config={
        "dataset_name":args.dataset_name,
        "source":args.source
    }
    app(**app_config)