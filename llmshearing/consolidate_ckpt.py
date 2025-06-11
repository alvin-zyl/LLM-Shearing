import sys
from llmshearing.models.composer_llama import ComposerMosaicLlama
from llmshearing.models.composer_cola import ComposerMosaicCoLA
import torch
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import FileSystemReader
from omegaconf import OmegaConf as om


def main(cfg):
    print("Building model")
    if "cola" in cfg.model.name or cfg.model.cola_module is not None:
        model = ComposerMosaicCoLA(cfg.model)
    else:
        model = ComposerMosaicLlama(cfg.model)
    print("Model built")

    if isinstance(model, ComposerMosaicCoLA) and cfg.consolidate_cola_params_only:
        print("Consolidating cola params only")
        state_dict = {
            "state": {"model": {k: v for k, v in model.state_dict().items() if "cola" in k}}
        }
    else:
        state_dict = {"state": {"model": model.state_dict()}}

    print("Loading from sharded checkpoint")
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(cfg.load_path),
        no_dist=True,
    )
    print("Loading finished")

    save_path = (
        f"{cfg.load_path}.pt"
        if not cfg.consolidate_cola_params_only
        else f"{cfg.load_path}_cola_params.pt"
    )
    print(f"Saving as full to {save_path}")
    torch.save(state_dict, save_path)
    print("Saved")


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    main(cfg)
