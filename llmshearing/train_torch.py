# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import datetime
import os
import sys
import warnings
from types import MethodType
from typing import Any, Dict

import random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from loguru import logger
import wandb
from train_torch_utils import *

from composer.core import (
    Algorithm,
    AlgorithmPass,
    Batch,
    BreakEpochException,
    Callback,
    DataSpec,
    Engine,
    Evaluator,
    Event,
    Precision,
    PyTorchScheduler,
    Time,
    Timestamp,
    TimeUnit,
    TrainerMode,
    ensure_data_spec,
    ensure_evaluator,
    ensure_time,
    get_precision_context,
    validate_eval_automicrobatching,
)
from composer.devices import DeviceCPU, DeviceGPU
from composer import Logger, Trainer
from composer.callbacks.checkpoint_saver import CheckpointSaver
from composer.core import Evaluator, Event
from composer.loggers import FileLogger
from composer.optim import DecoupledAdamW
from composer.utils import get_device, reproducibility
from llmfoundry.optim import (
    DecoupledAdaLRLion,
    DecoupledClipLion,
    DecoupledLionW,
    DecoupledLionW_8bit,
)
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_logger,
    build_scheduler,
)
from llmfoundry.utils.config_utils import log_config, pop_config, update_batch_size_info
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch import nn
import torch.distributed
from torch.optim.optimizer import Optimizer

from llmshearing.callbacks.callbacks import DebugCallback
from llmshearing.callbacks.dynamic_loading_callback import DynamicLoadingCallback
from llmshearing.callbacks.pruning_callback import PruningCallback
from llmshearing.datasets.load_text_dataloader import build_text_dataloader
from llmshearing.models.model_registry import COMPOSER_MODEL_REGISTRY
from state import State

import streaming
streaming.base.util.clean_stale_shared_memory()


def is_one_hour(run_name: str):
    """Check if the run name is for one hour training."""
    return run_name.startswith("ONE_HOUR")


def exit_batch_checkpoint(self, state: State, logger: Logger):
    """Exit the program after saving the checkpoint."""
    if (
        self.save_interval(state, Event.BATCH_CHECKPOINT)
        and self.last_checkpoint_batch != state.timestamp.batch
    ):
        self._save_checkpoint(
            state,
            logger,
        )
        print("Ending program at batch", state.timestamp.batch)
        print(self.folder)
        sys.exit()


def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        loaders.append(cfg.eval_loader)


def build_composer_model(cfg: DictConfig):
    """build the composer model"""
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    return COMPOSER_MODEL_REGISTRY[cfg.name](cfg)


def load_weights(cfg: DictConfig):
    """load weights"""
    if cfg.model.get("path", None):
        state_dict = torch.load(cfg.model.path)  # for loading pre-trained llama
        if "state" in state_dict:
            state_dict = state_dict["state"]["model"]
        logger.info(f"Loaded model from path: {cfg.model.path}")
        return state_dict
    return None


def load_state_dict(model: nn.Module, state_dict: Dict[str, Any]):
    """load state dict to the model"""
    result = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Model load state dict result: {result}")
    logger.info("Having missing rotary_emb.inv_freq keys is normal")


def build_optimizer(
    model: torch.nn.Module, name: str, optimizer_config: Dict[str, Any]
) -> Optimizer:
    """
    build optimizer that consists of three groups of parameters:
    - main_model_params: parameters of the main model
    - l0_module_params: parameters of the l0 module
    - lagrange_params: parameters of the lagrange multipliers
    """
    param_groups = {}
    main_model_params = [p for n, p in model.named_parameters() if "l0_module" not in n]
    l0_module_params = [
        p for n, p in model.named_parameters() if "l0_module" in n and "lambda" not in n
    ]
    lagrange_params = [
        p for n, p in model.named_parameters() if "l0_module" in n and "lambda" in n
    ]

    param_groups = [{"params": main_model_params, "lr": optimizer_config.lr}]
    lag_lr = pop_config(optimizer_config, "lag_lr")
    if len(l0_module_params) > 0:
        param_groups.extend(
            [
                {"params": l0_module_params, "lr": lag_lr},
                {"params": lagrange_params, "lr": -(lag_lr)},
            ]
        )

    for i, group in enumerate(param_groups):
        logger.info(
            f"Group {i}:",
            f"{len(group['params'])} tensors",
            f"{sum(p.numel() for p in group['params'])} params",
            f"{group['lr']:.2e} lr",
        )

    if name == "decoupled_adamw":
        return DecoupledAdamW(param_groups, **optimizer_config)
    elif name == "decoupled_lionw":
        return DecoupledLionW(param_groups, **optimizer_config)
    elif name == "clip_lion":
        return DecoupledClipLion(param_groups, **optimizer_config)
    elif name == "adalr_lion":
        return DecoupledAdaLRLion(param_groups, **optimizer_config)
    elif name == "decoupled_lionw_8b":
        return DecoupledLionW_8bit(param_groups, **optimizer_config)
    else:
        raise ValueError(f"Not sure how to build optimizer: {name}")


def main(cfg):
    """Main training function"""
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=f"torch.distributed.*_base is a private function and will be deprecated.*",
    )

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID")))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"[{os.getpid()}] Start running: Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
    )

    device = DeviceGPU(local_rank)
    rank_zero_seed = distribute_and_get_random_seed(cfg.seed, device)
    seed = rank_zero_seed + global_rank
    logger.info(f"Rank zero seed: {rank_zero_seed}, rank {global_rank} seed: {seed}")
    reproducibility.seed_all(seed)

    # Get batch size info
    cfg = update_batch_size_info(cfg)
    device_train_microbatch_size = cfg.get("device_train_microbatch_size", "auto")
    auto_microbatching = is_auto_microbatching(
        device_train_microbatch_size, device=device
    )
    # Dataloaders
    if global_rank == 0:
        logger.info("Building train loader...")
    train_loader = build_text_dataloader(
        cfg.train_loader,
        cfg.device_train_batch_size,
        cfg.callbacks.data_loading.dynamic,
        cfg.callbacks.data_loading.set_names,
        proportion=cfg.callbacks.data_loading.proportion,
    )

    if global_rank != 0:
        logger.remove()

    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    # Run Name
    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("RUN_NAME", "Debug")

    # Read FSDP Config as a dict
    fsdp_config = cfg.get("fsdp_config", None)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

    # Restrict model init_device to 'meta' and 'cpu',
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_device = cfg.model.get("init_device", "cpu")
    assert init_device in ["meta", "cpu"]
    if fsdp_config is None and init_device == "meta":
        warnings.warn(
            "Using `cfg.model.init_device='meta'` is only valid when using FSDP! "
            + "Reverting to `cfg.model.init_device='cpu'`."
        )
        cfg.model.init_device = "cpu"

    save_folder = cfg.save_folder.replace("{run_name}", cfg.run_name)
    filename = f"{save_folder}/logs.txt"
    count = 1

    while os.path.exists(filename):
        logger.info(f"File {filename} already exists")
        filename = f"{save_folder}/logs_{count}.txt"
        count += 1

    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get("loggers") or {}).items()
    ]
    loggers.append(FileLogger(filename=filename, buffer_size=1, flush_interval=50))
    if global_rank == 0:
    #     loggers.append(FileLogger(filename=filename, buffer_size=1, flush_interval=50))
        logger.info(f"Logging to {filename}")
        logger.add(filename)

    # Build Model
    logger.info("Initializing model...")
    if cfg.callbacks.data_loading.dynamic:
        cfg.model.set_names = cfg.callbacks.data_loading.set_names
    model = build_composer_model(cfg.model)
    logger.info(model)
    logger.info(cfg.model.l0_module)

    state_dict = load_weights(cfg)
    if state_dict is not None:
        load_state_dict(model, state_dict)

    cfg.n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{cfg.n_params=:.2e}")
    if hasattr(model, "num_fwd_flops"):
        logger.info(f"{model.num_fwd_flops=:.2e}")

    # set names has to be part of the config
    assert (
        getattr(cfg.callbacks.data_loading, "set_names", None) is not None
    ), "please specify the set (domain) names in the config"

    # Optimizer
    optimizers = build_optimizer(model, cfg.optimizer.pop("name"), cfg.optimizer)

    model = device.module_to_device(model)
    optimizers = map_collection(optimizers, device.optimizer_to_device)

    # Scheduler
    schedulers = build_scheduler(cfg.scheduler.pop("name"), cfg.scheduler)

    # Callbacks
    callbacks = []
    data_loading_config = pop_config(cfg.callbacks, "data_loading")
    if data_loading_config.dynamic:
        dl_callback = DynamicLoadingCallback(
            target_loss=data_loading_config.target_loss,
            proportion=data_loading_config.proportion,
            set_names=data_loading_config.set_names,
            update_type=data_loading_config.update_type,
        )
        callbacks.append(dl_callback)
    callbacks += [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get("callbacks") or {}).items()
    ]
    if model.model.l0_module is not None:  # pruning callback
        callbacks.append(PruningCallback(save_folder=cfg.save_folder))

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get("algorithms") or {}).items()
    ]

    precision = cfg.precision
    if precision is None:
        precision = (
            Precision.AMP_FP16 if isinstance(device, DeviceGPU) else Precision.FP32
        )
    elif isinstance(precision, str):
        precision = Precision(precision)
    validate_precision(precision, device)

    state = State(
        rank_zero_seed=rank_zero_seed,
        algorithms=algorithms,
        model=model,
        device=device,
        callbacks=callbacks,
        device_train_microbatch_size=device_train_microbatch_size,
        auto_microbatching=auto_microbatching,
        precision=precision,
        optimizers=optimizers,
        run_name=cfg.run_name,
        fsdp_config=set_fsdp_default(fsdp_config) if fsdp_config is not None else None,
    )
    engine = CallbackEngine()

    state.callbacks[:] = list(cast(List[Callback], loggers)) + state.callbacks
    
    loggers = Logger(state=state, destinations=loggers)
    logger.info("Logging config...")
    logger.info(cfg)
    if global_rank == 0:
        wandb.init(project="pruning", name=cfg.run_name)

    state.model.logger = loggers
    engine.init(state, loggers)

    state.train_metrics = deepcopy(state.model.get_metrics(is_train=True))
    state.eval_metrics = {}

    if train_loader is not None:
        train_data_spec = ensure_data_spec(train_loader)
        state.train_data_spec = train_data_spec
        state.set_dataloader(train_data_spec.dataloader, "train")
        state.train_dataloader = state.dataloader
        state.device_train_microbatch_size = get_initial_device_train_microbatch_size(
            state.device_train_microbatch_size,
            state.auto_microbatching,
            state.train_dataloader,
        )

    state.max_duration = ensure_time(cfg.max_duration, TimeUnit.EPOCH)
    state.schedulers = compile_schedulers(schedulers, state, 1.0)
    scheduler_step_frequency = TimeUnit.BATCH

    backwards_create_graph = any((x.backwards_create_graph for x in state.algorithms))
    state.backwards_create_graph = backwards_create_graph
    find_unused_parameters = any((x.find_unused_parameters for x in state.algorithms))
    state.find_unused_parameters = find_unused_parameters
    ddp_sync_strategy = get_ddp_sync_strategy(None, find_unused_parameters=find_unused_parameters)
    state.ddp_sync_strategy = ddp_sync_strategy
    warnings.filterwarnings(action="ignore", message="torch.cuda.amp.GradScaler")
    state.scaler = ClosureGradScaler() if use_closures(state) else GradScaler()

    if state.fsdp_config is not None:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        state.scaler = ShardedGradScaler()

    state._original_model = state.model
    if state.fsdp_config is not None and not state.load_fsdp_monolith_rank0_only:
        prepare_fsdp_module(
            model, optimizers, state.fsdp_config, precision, device, auto_microbatching
        )

    reproducibility.seed_all(state.seed)

    if not state.fsdp_enabled and dist.get_world_size() > 1:
        state.model = prepare_ddp_module(state.model, find_unused_parameters)

    engine.fit_start(state, loggers)
    use_grad_scaling = use_grad_scaling_(state, state.precision, state.scaler)
    spin_dataloaders_to_cur_epoch(state)
    state.model.train()
    finished_epoch_early = False
    last_wct = datetime.datetime.now()

    while state.timestamp < state.max_duration:
        if int(state.timestamp.batch_in_epoch) == 0:
            engine.epoch_start(state, loggers)
            loggers.log_metrics({'time/epoch': state.timestamp.epoch.value})
        
        dataloader = state.dataloader
        if isinstance(dataloader, DataLoader) and isinstance(
            dataloader.sampler, DistributedSampler
        ):
            dataloader.sampler.set_epoch(int(state.timestamp.epoch))

        loggers.log_metrics({'time/epoch': state.timestamp.epoch.value})

        for batch_idx, state.batch in enumerate(iter_dataloader(state, engine, loggers)):
            state.batch = state.device.batch_to_device(state.batch)
            state.batch = state.train_data_spec.device_transforms(state.batch)
            rank_num_samples = state.train_data_spec.get_num_samples_in_batch(state.batch)
            rank_num_tokens = state.train_data_spec.get_num_tokens_in_batch(state.batch)

            engine.after_dataloader(state, loggers)
            engine.batch_start(state, loggers)

            loggers.log_metrics({
                'time/batch': state.timestamp.batch.value,
                'time/sample': state.timestamp.sample.value,
                'time/batch_in_epoch': state.timestamp.batch_in_epoch.value,
                'time/sample_in_epoch': state.timestamp.sample_in_epoch.value,
            })

            if rank_num_tokens > 0:
                loggers.log_metrics({'time/token': state.timestamp.token.value})
                loggers.log_metrics({'time/token_in_epoch': state.timestamp.token_in_epoch.value})

            total_loss_dict = train_batch(state, engine, use_grad_scaling, loggers)

            if use_grad_scaling:
                state.scaler.update()

            # total_loss_dict can be None if gradient scaling failed
            if total_loss_dict is not None:
                map_collection(total_loss_dict, dist.all_reduce)
                total_loss_dict = {
                    k: loss.cpu().item() / dist.get_world_size() for k, loss in total_loss_dict.items()
                }
                state.total_loss_dict = total_loss_dict
                loggers.log_metrics(total_loss_dict)

            # The scheduler step.step() and compute_and_log_metrics() are going to be included in the
            # next batch's wall clock time. The time accumulation must be done here so schedulers
            # have the latest timing information

            now = datetime.datetime.now()

            batch_time = now - last_wct

            total_num_samples, total_num_tokens, batch_time = accumulate_time_across_ranks(
                state,
                rank_num_samples,
                rank_num_tokens,
                batch_time,
            )

            # `now` is actually in the past, but want to include the time it takes to perform this reduction
            last_wct = now

            if scheduler_step_frequency == TimeUnit.BATCH:
                for scheduler in state.schedulers:
                    scheduler.step()

            if state.train_metrics is not None:
                compute_and_log_metrics(
                    state,
                    dataloader_label='train',
                    metrics=state.train_metrics,
                    logger=loggers
                )

            state.previous_timestamp = state.timestamp
            state.timestamp = state.timestamp.to_next_batch(
                samples=total_num_samples,
                tokens=total_num_tokens,
                duration=batch_time,
            )

            engine.batch_end(state, loggers)

            # Pause the timing during evaluation
            # Evaluation time is tracked separately in state.eval_timestamp
            duration = datetime.datetime.now() - last_wct
            run_evaluators(state, Event.BATCH_END, engine, loggers)
            last_wct = datetime.datetime.now() - duration

            engine.batch_checkpoint(state, loggers)

            if state.timestamp >= state.max_duration:
                # If max_duration is specified in batches, samples, or tokens, and
                # and the max_duration is reached mid-epoch, then break out of the dataloader
                # to finish the epoch early and finish training.
                finished_epoch_early = True
                break

        if not finished_epoch_early or state.dataloader_len == state.timestamp.batch_in_epoch:
            # Trigger the epoch end events if the dataloader was exhausted.
            # This happens if the "break" did not trigger above, or if it
            # did (e.g. duration specified in samples/batches/tokens), but it is still
            # the end of the dataloader (i.e. next(dataloader) would raise StopIteration)
            if state.train_metrics is not None:
                state.train_metrics = ensure_metrics_device_and_dtype(state, state.train_metrics)
                compute_and_log_metrics(
                    state, 
                    dataloader_label='train',
                    metrics=state.train_metrics,
                    logger=loggers
                )

            if scheduler_step_frequency == TimeUnit.EPOCH:
                for scheduler in state.schedulers:
                    scheduler.step()

            state.previous_timestamp = state.timestamp
            state.timestamp = state.timestamp.to_next_epoch()

            engine.epoch_end(state, loggers)

            # Pause the timing during evaluation
            # Evaluation time is tracked separately in state.eval_timestamp
            duration = datetime.datetime.now() - last_wct
            run_evaluators(state, Event.EPOCH_END, engine, loggers)
            last_wct = datetime.datetime.now() - duration

            engine.epoch_checkpoint(state, loggers)

    loggers.log_metrics({
        'time/epoch': state.timestamp.epoch.value,
        'time/batch': state.timestamp.batch.value,
        'time/sample': state.timestamp.sample.value,
        'time/batch_in_epoch': state.timestamp.batch_in_epoch.value,
        'time/sample_in_epoch': state.timestamp.sample_in_epoch.value,
    })
    if state.previous_timestamp is not None and state.timestamp.token.value - state.previous_timestamp.token.value > 0:
        loggers.log_metrics({'time/token_in_epoch': state.timestamp.token_in_epoch.value})
        loggers.log_metrics({'time/token': state.timestamp.token.value})

    engine.fit_end(state, loggers)
    run_evaluators(state, Event.FIT_END, engine, loggers)

    print("Done.")


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)

    # save the config files
    save_dir = cfg.save_folder.replace("{run_name}", cfg.run_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(cfg, save_dir + "/config.pt")

    main(cfg)
