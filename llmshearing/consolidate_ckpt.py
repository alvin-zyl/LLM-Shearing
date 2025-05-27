import sys
from llmshearing.models.composer_llama import ComposerMosaicLlama
import torch
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import FileSystemReader
from omegaconf import OmegaConf as om


def main(cfg):
    print("Building model")
    model = ComposerMosaicLlama(cfg.model)
    print("Model built")

    state_dict = {
        "state": {"model": model.state_dict()}
    }

    print("Loading from sharded checkpoint")
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(cfg.load_path),
        no_dist=True,
    )
    print("Loading finished")

    print(f"Saving as full to {cfg.load_path}.pt")
    torch.save(state_dict, f"{cfg.load_path}.pt")
    print("Saved")



if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    main(cfg)
