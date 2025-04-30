import contextlib
from copy import deepcopy
import datetime
import itertools
import re
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
    List,
    Sequence,
    Iterable,
    Callable,
    ContextManager,
)
import warnings
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler
from loguru import logger
import math, torch
import collections.abc
from omegaconf import DictConfig
from composer import Callback
from composer.utils import get_device, reproducibility
from composer.devices import Device, DeviceGPU, DeviceCPU, DeviceMPS
from composer.core import (
    Precision,
    PyTorchScheduler,
    TrainerMode,
    DataSpec,
    get_precision_context,
    Event,
    Evaluator,
    Timestamp,
    Time,
)
from composer.trainer.dist_strategy import prepare_ddp_module, prepare_fsdp_module
from composer.trainer._scaler import ClosureGradScaler
from torch.cuda.amp.grad_scaler import GradScaler
from composer.optim import ComposerScheduler, compile_composer_scheduler
from composer.trainer._scale_schedule import scale_pytorch_scheduler
from composer.core.data_spec import ensure_data_spec
from composer.loggers import Logger
from composer.core.types import Batch
from composer.trainer.dist_strategy import DDPSyncStrategy, ddp_sync_context
from state import State
from torchmetrics import Metric
from composer.utils.misc import is_model_deepspeed, model_eval_mode

Scheduler = Union[ComposerScheduler, PyTorchScheduler]


def calculate_batch_size_info(
    global_batch_size: int, device_microbatch_size: Union[int, Literal["auto"]]
) -> Tuple[int, Union[int, Literal["auto"]], Union[int, Literal["auto"]]]:
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            + "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            + f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == "auto":
        device_grad_accum = "auto"
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            logger.warning(
                f"device_microbatch_size > device_batch_size, "
                + f"will be reduced from {device_microbatch_size} -> {device_batch_size}."
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size / device_microbatch_size)
    else:
        raise ValueError(f"Not sure how to parse {device_microbatch_size=}")

    return device_batch_size, device_microbatch_size, device_grad_accum


def update_batch_size_info(cfg: DictConfig) -> DictConfig:
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = (
        calculate_batch_size_info(
            cfg.global_train_batch_size, cfg.device_train_microbatch_size
        )
    )
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if "device_eval_batch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def distribute_and_get_random_seed(seed: Optional[int], device: Device):
    if seed is None:
        seed = reproducibility.get_random_seed()

    # Ensure that each process has a seed = rank_zero_seed + global_rank
    # This "deterministically different" seed behavior is required to be able
    # to restore seeds when resuming form checkpoints, since only the
    # `rank_zero_seed` is stored on state.
    if seed < 0 or seed > reproducibility.MAX_SEED:
        raise ValueError(f"Invalid seed: {seed}. It must be on [0; 2**32 - 1)")

    # using int64 to prevent overflow
    rank_zero_seed = device.tensor_to_device(torch.tensor([seed], dtype=torch.int64))
    if dist.get_world_size() > 1:
        dist.broadcast(rank_zero_seed, src=0)
    rank_zero_seed = rank_zero_seed.item()
    assert isinstance(rank_zero_seed, int)
    return rank_zero_seed


def is_auto_microbatching(
    device_train_microbatch_size: Optional[Union[int, str]], device: Device
):
    if device_train_microbatch_size == "auto":
        logger.warning(
            (
                "`device_train_microbatch_size='auto'` may potentially fail with unexpected "
                "CUDA errors. Auto microbatching attempts to catch CUDA Out of Memory errors "
                "and adjust the batch size, but it is possible CUDA will be put into an "
                "irrecoverable state due to PyTorch bugs, e.g. integer overflow. In this case, "
                "please manually set device_train_microbatch_size explicitly to an integer "
                "instead."
            )
        )
        if not isinstance(device, DeviceGPU):
            raise ValueError(
                "Can only use adaptive device_train_microbatch_size on GPU. Please set device_train_microbatch_size >= 1."
            )
        return True
    else:
        return False


def filter_metrics(
    metrics: Dict[str, Metric], metric_names: Optional[List[str]]
) -> Dict[str, Metric]:
    """
    Filter the metrics based on the given metric_names as regex strings (e.g. 'Accuracy', 'f1' for 'BinaryF1Score',
    'Top-.' for 'Top-1 Accuracy' and 'Top-2 Accuracy', etc).
    If no metric_names are provided, all metrics will be returned.
    """
    metrics = deepcopy(metrics)
    if not metric_names:
        return metrics
    else:
        filtered_metrics = {}
        for name, metric in metrics.items():
            if any(
                re.match(f".*{metric_name}.*", name, re.IGNORECASE)
                for metric_name in metric_names
            ):
                filtered_metrics[name] = metric
        return filtered_metrics


def set_evaluator_interval_and_subset_num_batches(
    evaluators: Sequence[Evaluator],
    eval_interval: Union[int, str, Time, Callable[[State, Event], bool]],
    subset_num_batches: int,
):
    # convert eval_dataloader to `List[Evaluator]`
    for evaluator in evaluators:
        if evaluator.subset_num_batches is None:
            evaluator.subset_num_batches = subset_num_batches
        if evaluator.eval_interval is None:
            evaluator.eval_interval = eval_interval
        eval_dataloader = evaluator.dataloader.dataloader
        if isinstance(eval_dataloader, collections.abc.Sized) and (
            evaluator.subset_num_batches is None or evaluator.subset_num_batches == -1
        ):
            try:
                dataloader_len = len(eval_dataloader)
            except TypeError:
                dataloader_len = None
            if dataloader_len == None:
                raise ValueError(
                    "eval_subset_num_batches must be set when using an infinite sized "
                    "eval_dataloader where length is `None`. Otherwise, evaluation will "
                    "run forever and never terminate."
                )


def validate_precision(precision: Precision, device: Device):
    if isinstance(device, DeviceCPU) and precision != Precision.FP32:
        raise ValueError(f"{precision} is not supported for CPU training.")


def map_collection(collection, map_fn):
    """Applies ``map_fn`` on each element in ``collection``.

    * If ``collection`` is a tuple or list of elements, ``map_fn`` is applied on each element,
      and a tuple or list, respectively, containing mapped values is returned.
    * If ``collection`` is a dictionary, ``map_fn`` is applied on each value, and a dictionary
      containing the mapped values is returned.
    * If ``collection`` is ``None``, ``None`` is returned.
    * If ``collection`` is a single element, the result of applying ``map_fn`` on it is returned.

    Args:
        collection: The element, or a tuple of elements.
        map_fn: A function to invoke on each element.

    Returns:
        Collection: The result of applying ``map_fn`` on each element of ``collection``.
        The type of ``collection`` is preserved.
    """
    if collection is None:
        return None
    if isinstance(collection, (tuple, list)):
        return type(collection)(map_fn(x) for x in collection)
    if isinstance(collection, dict):
        return {k: map_fn(v) for k, v in collection.items()}
    return map_fn(collection)


def set_fsdp_default(fsdp_config: Dict[str, Any]):
    """Modify fsdp_config to set default values for missing keys."""
    fsdp_config.setdefault("activation_checkpointing", False)
    fsdp_config.setdefault("activation_checkpointing_reentrant", True)
    fsdp_config.setdefault("activation_cpu_offload", False)
    fsdp_config.setdefault("backward_prefetch", "BACKWARD_POST")
    fsdp_config.setdefault("cpu_offload", False)
    fsdp_config.setdefault("flatten_parameters", True)
    fsdp_config.setdefault("forward_prefetch", False)
    fsdp_config.setdefault("ignored_modules", None)
    fsdp_config.setdefault("keep_low_precision_grads", False)
    fsdp_config.setdefault("limit_all_gathers", False)
    fsdp_config.setdefault("load_monolith_rank0_only", False)
    fsdp_config.setdefault("mixed_precision", "DEFAULT")
    fsdp_config.setdefault("sharded_ckpt_prefix_dir", "ep{epoch}-ba{batch}")
    fsdp_config.setdefault("sharding_strategy", "FULL_SHARD")
    fsdp_config.setdefault("state_dict_type", "full")
    fsdp_config.setdefault("sync_module_states", False)
    fsdp_config.setdefault("use_orig_params", True)
    fsdp_config.setdefault("verbose", False)
    return fsdp_config


def use_closures(state: State) -> bool:
    """Determines based on precision and optimizers whether to use closures.

    We default to using closures unless AMP is enabled, in which case we only allow closures when using optimizers
    with the _step_supports_amp_closure flag.
    """
    if state.deepspeed_enabled:
        return False

    if state.precision != Precision.AMP_FP16:
        return True

    if state.optimizers is None:
        raise RuntimeError(
            "state.optimizers must be set before `_use_closures` can be determined"
        )

    return all(
        getattr(optimizer, "_step_supports_amp_closure", False)
        for optimizer in ensure_tuple(state.optimizers)
    )


def ensure_tuple(x):
    """Converts ``x`` into a tuple.

    * If ``x`` is ``None``, then ``tuple()`` is returned.
    * If ``x`` is a tuple, then ``x`` is returned as-is.
    * If ``x`` is a list, then ``tuple(x)`` is returned.
    * If ``x`` is a dict, then ``tuple(v for v in x.values())`` is returned.

    Otherwise, a single element tuple of ``(x,)`` is returned.

    Args:
        x (Any): The input to convert into a tuple.

    Returns:
        tuple: A tuple of ``x``.
    """
    if x is None:
        return ()
    if isinstance(x, (str, bytes, bytearray)):
        return (x,)
    if isinstance(x, collections.abc.Sequence):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    return (x,)


def cast(typ, val):
    """Cast a value to a type.

    This returns the value unchanged.  To the type checker this
    signals that the return value has the designated type, but at
    runtime we intentionally don't check anything (we want this
    to be as fast as possible).
    """
    return val


def compile_schedulers(
    schedulers: Optional[Union[Scheduler, Sequence[Scheduler]]],
    state: State,
    scale_schedule_ratio: float,
) -> List[PyTorchScheduler]:
    compiled_schedulers = []
    for scheduler in ensure_tuple(schedulers):
        if isinstance(scheduler, PyTorchScheduler):
            scale_pytorch_scheduler(scheduler, scale_schedule_ratio)
            compiled_schedulers.append(scheduler)
        else:  # it's a composer scheduler
            compiled_schedulers.append(
                compile_composer_scheduler(scheduler, state, scale_schedule_ratio)
            )

    return compiled_schedulers


def get_initial_device_train_microbatch_size(
    device_train_microbatch_size: Optional[Union[int, str]],
    auto_microbatching: bool,
    train_dataloader: Optional[Iterable],
) -> Optional[int]:
    """Sets initial value of device_train_microbatch_size.

    If auto_microbatching, sets initial `device_train_microbatch_size` to per rank batch size. If
    `train_dataloader` is not set yet, returns None and this function will be called again when
    `train_dataloader` is set, such as when `fit()` is called.
    """
    if device_train_microbatch_size is None or auto_microbatching:
        # Return None, this function will be called again when `train_dataloader` is set
        if train_dataloader is None:
            return None
        try:
            batch_size = getattr(train_dataloader, "batch_size")
        except AttributeError as e:
            # Error message when `device_train_microbatch_size` is None
            # Note: This code path will be removed after `auto` is made default
            if device_train_microbatch_size is None:
                raise ValueError(
                    "`device_train_microbatch_size` must be set when `state.train_dataloader` does not have a `batch_size` attribute."
                ) from e
            # Error message when `device_train_microbatch_size` is 'auto'
            raise AttributeError(
                "`device_train_microbatch_size='auto'` requires the `state.train_dataloader` to have a `batch_size` attribute."
            ) from e
        return batch_size
    elif isinstance(device_train_microbatch_size, int):
        return device_train_microbatch_size
    else:
        raise ValueError("device_train_microbatch_size must be an int or ``'auto'``")


def iter_dataloader(state: State, engine: "CallbackEngine", logger: Logger):
    """Helper method to iterate over the dataloader.

    This method yields up to :attr:`.State.dataloader_len`` batches from the dataloader. In addition, if the
    profiler is enabled, the dataloader latency recorded via the :class:`.Marker` API.

    Args:
        trainer_mode (TrainerMode): Specifies which mode the trainer is in.
    """
    assert (
        state.dataloader is not None
    ), "the dataloader should be set before calling this method"

    if state.dataloader_len is None:
        dataloader_iter = iter(state.dataloader)
    else:
        dataloader_iter = itertools.islice(state.dataloader, int(state.dataloader_len))

    while True:
        try:
            engine.before_dataloader(state, logger)
            batch = next(dataloader_iter)
        except StopIteration:
            break
        
        yield batch


def use_grad_scaling_(
    state: State, precision: Union[str, Precision], scaler: Optional[GradScaler]
) -> bool:
    """Determines based on precision when to use grad scaling.

    By default, the pytorch GradScaler is a no-op if running on
    unsupported hardware. Here we raise a RuntimeError instead.

    Args:
        precision (Precision): Numerical precision, based on the Precision Enum.
        scaler (GradScaler): Used to make sure that the scaler is enabled when
        using grad scaling.

    Raises:
        RuntimeError:
            Occurs when attempting to use grad scaling without the scaler
            enabled. Likely due to hardware not supporting the provided precision.
    """
    if state.deepspeed_enabled:
        return False

    precision = Precision(precision)
    use_grad_scaling = precision == Precision.AMP_FP16

    if use_grad_scaling and (scaler is None or not scaler.is_enabled()):
        raise RuntimeError(
            f"Attempting to use grad scaling with {precision}, but scaler is not enabled."
            f"Potentially your hardware does not support Precision {precision}."
        )
    return use_grad_scaling


def spin_dataloaders_to_cur_epoch(state: State):
    """Spin the dataloaders to restore sampler state for current epoch.

    Only one batch must be loaded to seed the sampler's generator. since only the first batch is being loaded, the
    dataloader may not be completely iterated through.
    """
    # spin the evaluator dataloaders once to initialize its sampler deterministically
    # so it does not affect any other RNG reads
    eval_state = state.dataset_resumption.get("eval", {})
    for evaluator in state.evaluators:
        dataloader = evaluator.dataloader.dataloader
        if isinstance(dataloader, DataLoader) and isinstance(
            dataloader.sampler, DistributedSampler
        ):
            dataloader.sampler.set_epoch(0)
        if evaluator.label not in eval_state:
            for _ in dataloader:
                break

    # spin the train dataloader's sampler to get to the state of the desired epoch
    dataloader = state.dataloader
    assert dataloader is not None, "train dataloader is set on state after FIT_START"
    if "train" not in state.dataset_resumption:
        for epoch in range(int(state.timestamp.epoch)):
            if isinstance(dataloader, DataLoader) and isinstance(
                dataloader.sampler, DistributedSampler
            ):
                dataloader.sampler.set_epoch(epoch)
            for _ in dataloader:
                break


def get_ddp_sync_strategy(
    ddp_sync_strategy: Optional[Union[str, DDPSyncStrategy]],
    find_unused_parameters: bool,
):
    if ddp_sync_strategy is None:
        if find_unused_parameters:
            ddp_sync_strategy = DDPSyncStrategy.MULTI_AUTO_SYNC
        else:
            ddp_sync_strategy = DDPSyncStrategy.SINGLE_AUTO_SYNC
    else:
        ddp_sync_strategy = DDPSyncStrategy(ddp_sync_strategy)
    return ddp_sync_strategy


def _get_precision_context(
    precision: Precision,
    precision_config: Optional[Dict[str, Any]],
    deepspeed_enabled: bool,
):
    if deepspeed_enabled:
        return contextlib.nullcontext()
    return get_precision_context(precision, precision_config)


class CallbackEngine:

    def __init__(self):
        pass

    def init(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.init(state, logger)

    def after_load(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_load(state, logger)

    def fit_start(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.fit_start(state, logger)

    def epoch_start(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.epoch_start(state, logger)

    def before_dataloader(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.before_dataloader(state, logger)

    def after_dataloader(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_dataloader(state, logger)

    def batch_start(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.batch_start(state, logger)

    def before_train_batch(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.before_train_batch(state, logger)

    def before_forward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.before_forward(state, logger)

    def after_forward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_forward(state, logger)

    def before_loss(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.before_loss(state, logger)

    def after_loss(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_loss(state, logger)

    def before_backward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.before_backward(state, logger)

    def after_backward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_backward(state, logger)

    def after_train_batch(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.after_train_batch(state, logger)

    def batch_end(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.batch_end(state, logger)

    def batch_checkpoint(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.batch_checkpoint(state, logger)

    def epoch_end(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.epoch_end(state, logger)

    def epoch_checkpoint(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.epoch_checkpoint(state, logger)

    def eval_before_all(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_before_all(state, logger)

    def eval_start(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_start(state, logger)

    def eval_batch_start(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_batch_start(state, logger)

    def eval_before_forward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_before_forward(state, logger)

    def eval_after_forward(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_after_forward(state, logger)

    def eval_batch_end(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_batch_end(state, logger)

    def eval_end(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_end(state, logger)

    def eval_after_all(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.eval_after_all(state, logger)

    def fit_end(self, state: State, logger: Logger):
        for callback in state.callbacks:
            callback.fit_end(state, logger)


def ensure_metrics_device_and_dtype(state: State, metrics: Dict[str, Metric]):
    for name, metric in metrics.items():
        # Safety check to ensure the metric and data are on the same device. Normally not
        # needed because the metric is automatically on the same device as the model.
        # See https://torchmetrics.readthedocs.io/en/latest/pages/overview.html for details.
        metrics[name] = state.device.module_to_device(metric)
        if is_model_deepspeed(state.model):
            # HACK: DeepSpeed somehow manages to convert metric internal states to its own dtype. When
            # running with FP16, this tends to result in overflows. Let's assume FP32 is good enough.
            for key in metric._defaults:
                metric_data = getattr(metric, key)
                if (
                    isinstance(metric_data, torch.Tensor)
                    and metric_data.dtype == torch.float16
                ):
                    metric_data = metric_data.to(torch.float32)  # type: ignore
                    setattr(metric, key, metric_data)
    return metrics


def _eval_train_metrics(state: State, device_batch):
    assert (
        state.train_data_spec is not None
    ), "The train data spec should be set on __init__ or fit()"
    assert (
        state.train_metrics is not None
    ), "The train metrics should be set on __init__ or fit()"

    with torch.no_grad(), model_eval_mode(state.model), _get_precision_context(
        state.precision, state.precision_config, state.deepspeed_enabled
    ):
        eval_outputs = state._original_model.eval_forward(device_batch, state.outputs)
        for metric in state.train_metrics.values():
            state._original_model.update_metric(
                device_batch,
                eval_outputs,
                metric,
            )


def train_batch(
    state: State,
    engine: CallbackEngine,
    use_grad_scaling: bool,
    logger: Logger,
) -> Dict[str, torch.Tensor]:
    """Compute loss by training on a full batch of data.

    Adaptively change microbatch size if enabled to maximize GPU usage.

    Args:
        use_grad_scaling (bool): Enables gradient scaling.

    Returns:
        Dict[str, torch.Tensor]: a dictionary containing the total loss and individual losses if available.
    """
    assert (
        state.train_data_spec is not None
    ), "The train data spec should be set on __init__ or fit()"

    # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop.
    # Any in-place changes to a microbatch will be reflected in the device batch.
    device_batch = state.batch

    if state.train_metrics is not None:
        for metric in state.train_metrics.values():
            metric.reset()

    total_loss_dict = {
        "loss/train/total": state.device.tensor_to_device(torch.zeros(size=(1,)))
    }
    assert state.scaler is not None
    assert state.device_train_microbatch_size is not None
    microbatches = state.train_data_spec.split_batch(
        device_batch, state.device_train_microbatch_size
    )
    if use_closures(state):
        for optimizer in state.optimizers:
            if use_grad_scaling:
                state.scaler.step(
                    optimizer,
                    closure=lambda loss_dict=total_loss_dict, **kwargs: train_microbatches(
                        state,
                        engine,
                        logger,
                        use_grad_scaling,
                        microbatches,
                        loss_dict,
                        **kwargs,
                    ),
                )
            else:
                optimizer.step(
                    closure=lambda loss_dict=total_loss_dict, **kwargs: train_microbatches(
                        state,
                        engine,
                        logger,
                        use_grad_scaling,
                        microbatches,
                        loss_dict,
                        **kwargs,
                    ).item()
                )
    else:
        train_microbatches(
            state,
            engine,
            logger,
            use_grad_scaling,
            microbatches,
            total_loss_dict,
        )
        if not state.deepspeed_enabled:
            for optimizer in state.optimizers:
                if use_grad_scaling:
                    state.scaler.step(optimizer)
                else:
                    optimizer.step()

    if state.auto_microbatching:
        raise ValueError(f"auto_microbatching not supported yet")
    # Log microbatch and return loss if we've completed without OOMing.
    assert state.device_train_microbatch_size is not None
    logger.log_metrics(
        {"trainer/device_train_microbatch_size": state.device_train_microbatch_size}
    )
    return total_loss_dict


def train_microbatches(
    state: State,
    engine: CallbackEngine,
    logger: Logger,
    use_grad_scaling: bool,
    microbatches: Sequence[Batch],
    total_loss_dict: Dict[str, torch.Tensor],
    ddp_sync: bool = True,
) -> torch.Tensor:
    """Iterate over microbatches and compute the loss that will be used to step the optimizer.

    Args:
        microbatches (Sequence[Batch]): The microbatches which make up the batch.
        total_loss_dict (Dict[str, torch.tensor]): Dictionary containing individual losses and their sum aggregated across all
            microbatches.
        ddp_sync (bool): True to sync gradients between devices on every backwards
            pass and False to only sync gradients after each device has finished
            computing a gradient on it's entire set of microbatches. (default: ``True``)
    """

    if ddp_sync or not isinstance(state.model, DDP):
        context = contextlib.nullcontext
    else:
        context = cast(Callable[[], ContextManager], state.model.no_sync)

    assert state.train_data_spec is not None

    with context():
        engine.before_train_batch(state, logger)

        assert state.optimizers is not None
        assert state.scaler is not None

        if not state.deepspeed_enabled:
            for optimizer in state.optimizers:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except TypeError:
                    optimizer.zero_grad()

        # Tracker for gradient accumulation
        current_batch_size = sum(
            [
                state.train_data_spec.get_num_samples_in_batch(batch)
                for batch in microbatches
            ]
        )
        # Cache batch, which will be overwritten by microbatches. Restore after microbatches complete
        current_batch = state.batch

        for microbatch_idx, state.batch in enumerate(microbatches):
            is_final_microbatch = microbatch_idx + 1 == len(microbatches)
            microbatch_loss_dict = train_microbatch(
                state,
                engine,
                logger,
                use_grad_scaling,
                current_batch_size,
                is_final_microbatch,
            )

            # Aggregate each loss in microbatch_loss_dict into total_loss_dict
            for k, microbatch_loss in microbatch_loss_dict.items():
                loss_key = f"loss/train/{k}"
                if loss_key not in total_loss_dict:
                    total_loss_dict[loss_key] = state.device.tensor_to_device(
                        torch.zeros(size=(1,))
                    )
                total_loss_dict[loss_key] += microbatch_loss

        # Restore batch
        state.batch = current_batch

        # Unscale gradients before `Event.AFTER_TRAIN_BATCH`
        if use_grad_scaling:
            for optimizer in ensure_tuple(state.optimizers):
                state.scaler.unscale_(optimizer)

        engine.after_train_batch(state, logger)

        return total_loss_dict["loss/train/total"]


def train_microbatch(
    state: State,
    engine: CallbackEngine,
    logger: Logger,
    use_grad_scaling: bool,
    current_batch_size: int,
    is_final_microbatch: bool,
) -> Dict[str, torch.Tensor]:
    """Train and compute the loss of ``state.batch``, which is assumed to be a single microbatch.

    Args:
        use_grad_scaling (bool): Whether to use gradient scaling.
        current_batch_size (int): The current batch size.
        minibatch_num_samples (int): Number of samples in the minibatch.
        is_final_microbatch (bool): If current microbatch is the last one.
    """
    assert state.scaler is not None
    assert state.train_data_spec is not None

    # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop
    device_batch = deepcopy(state.batch)

    microbatch_num_samples = state.train_data_spec.get_num_samples_in_batch(state.batch)
    if state.deepspeed_enabled or not isinstance(state.model, DDP):
        sync_context = contextlib.nullcontext()
    else:
        sync_context = ddp_sync_context(
            state,
            is_final_microbatch,
            state.ddp_sync_strategy,
        )

    with sync_context:
        # Forward pass
        engine.before_forward(state, logger)

        with _get_precision_context(
            state.precision,
            state.precision_config,
            state.deepspeed_enabled,
        ):
            state.outputs = state.model(state.batch)

        engine.after_forward(state, logger)

        # Loss
        engine.before_loss(state, logger)

        with _get_precision_context(
            state.precision,
            state.precision_config,
            state.deepspeed_enabled,
        ):
            state.loss = state._original_model.loss(state.outputs, state.batch)

        assert state.loss is not None
        engine.after_loss(state, logger)

        # Backward Pass
        engine.before_backward(state, logger)

        microbatch_loss_dict = {}
        # If total loss key is present, copy loss
        if isinstance(state.loss, dict) and ("total" in state.loss):
            microbatch_loss = state.loss["total"]  # type: ignore
            microbatch_loss_dict = state.loss.copy()
        # If total loss key is not present, sum individual losses
        else:
            microbatch_loss = state.device.tensor_to_device(torch.zeros(size=(1,)))
            for loss in ensure_tuple(state.loss):
                microbatch_loss.add_(loss.mean())

            # Copy the loss if it is a dictionary
            if isinstance(state.loss, dict):
                microbatch_loss_dict = state.loss.copy()
            # If not, create a dictionary with generic loss names
            elif len(ensure_tuple(state.loss)) > 1:
                microbatch_loss_dict = {
                    f"loss{i}": loss for i, loss in enumerate(ensure_tuple(state.loss))
                }

            # Include total loss
            microbatch_loss_dict["total"] = microbatch_loss

        # For each loss to log: detach, clone, mean, then multiply by (microbatch size) / (batch size)
        for k, loss in microbatch_loss_dict.items():
            microbatch_loss_dict[k] = loss.detach().clone().mean() * (
                microbatch_num_samples / current_batch_size
            )

        if use_grad_scaling:
            microbatch_loss = cast(torch.Tensor, state.scaler.scale(microbatch_loss))

        if state.deepspeed_enabled:
            state.deepspeed_model.backward(microbatch_loss)

        else:
            # Scale loss based on the number of samples in the microbatch to maintain gradient numerics
            microbatch_loss.mul_(microbatch_num_samples / current_batch_size)
            microbatch_loss.backward(create_graph=state.backwards_create_graph)

        engine.after_backward(state, logger)

        # Use microbatch outputs to update training metrics
        if state.train_metrics is not None and len(state.train_metrics) != 0:
            state.train_metrics = ensure_metrics_device_and_dtype(
                state, state.train_metrics
            )
            _eval_train_metrics(state, device_batch)

    if state.deepspeed_enabled:
        state.deepspeed_model.step()

    return microbatch_loss_dict


def accumulate_time_across_ranks(
    state: State,
    num_samples: int,
    num_tokens: int,
    batch_time: datetime.timedelta,
) -> Tuple[int, int, datetime.timedelta]:
    """Accumulate the number of samples and tokens across ranks.

    Returns a (num_samples, num_tokens, batch_time) tuple.
    """
    # Samples and tokens should be summed
    # Batch time should be the value from rank 0
    sample_token_tensor = state.device.tensor_to_device(
        torch.tensor([num_samples, num_tokens], dtype=torch.int)
    )
    dist.all_reduce(sample_token_tensor, op=ReduceOp.SUM)
    batch_time_tensor = state.device.tensor_to_device(
        torch.tensor([batch_time.total_seconds()], dtype=torch.float32)
    )
    dist.broadcast(batch_time_tensor, src=0)
    batch_time = datetime.timedelta(seconds=batch_time_tensor[0].cpu().item())

    return (
        int(sample_token_tensor[0].cpu().item()),
        int(sample_token_tensor[1].cpu().item()),
        batch_time,
    )


def compute_and_log_metrics(
    state: State,
    dataloader_label: str,
    metrics: Dict[str, Metric],
    logger: Logger,
):
    """Computes metrics, logs the results, and updates the state with the deep-copied metrics.

    Args:
        dataloader_label (str): The dataloader label.
        metrics (Dict[str, Metric]): The metrics to compute.
    """
    metrics = deepcopy(metrics)

    # log computed metrics
    computed_metrics = {}
    for metric_name, metric in metrics.items():
        computed_metrics[metric_name] = metric.compute()

    logger.log_metrics(
        {
            f"metrics/{dataloader_label}/{name}": val
            for (name, val) in computed_metrics.items()
        },
    )

    # store metric instances
    for metric_name, metric in metrics.items():
        assert isinstance(metric, Metric)
        if dataloader_label == "train":
            state.train_metrics[metric_name] = metric
            state.train_metric_values[metric_name] = computed_metrics[metric_name]
        else:
            if dataloader_label not in state.eval_metrics:
                state.eval_metrics[dataloader_label] = {}
            state.eval_metrics[dataloader_label][metric_name] = metric
            state.eval_metric_values[metric_name] = computed_metrics[metric_name]


def run_evaluators(state: State, event: Event, engine: CallbackEngine, logger: Logger):
    """Runs evaluators periodically during training."""
    evaluators_executing = []
    for evaluator in state.evaluators:
        assert (
            evaluator.eval_interval is not None
        ), "eval_interval should have been set on __init__() or fit()"
        assert (
            evaluator.subset_num_batches is not None
        ), "subset_num_batches should have been set on __init__() or fit()"
        evaluators_executing.append(evaluator.eval_interval(state, event))
    if not any(evaluators_executing):
        return

    engine.eval_before_all(state, logger)
    for index, evaluator in enumerate(state.evaluators):
        if evaluators_executing[index]:
            _eval_loop(
                state,
                engine,
                evaluator=evaluator,
                subset_num_batches=evaluator.subset_num_batches,
                metrics=state.eval_metrics[evaluator.label],
                logger=logger,
            )

    engine.eval_after_all(state, logger)


def _eval_loop(
    state: State,
    engine: CallbackEngine,
    evaluator: Evaluator,
    metrics: Dict[str, Metric],
    logger: Logger,
    subset_num_batches: Optional[int] = None,
):
    """Evaluate the model and log appropriate metrics.

    Args:
        evaluator (Evaluator): The evaluator to use for evaluation.
        metrics (Dict[str, Metric]): Dictionary mapping metric names to metrics to evaluate against.
        subset_num_batches (int, optional): If specified, evaluate on this many batches. Defaults to ``-1``,
            which means to iterate over the entire dataloader.
    """
    if subset_num_batches is None:
        subset_num_batches = -1

    # back up the original dataloader on the state, so we can restore it after evaluation is finished
    original_dataloader = state.dataloader
    original_dataloader_label = state.dataloader_label
    original_num_batches = state.dataloader_len

    # Unpack data_spec
    data_spec = evaluator.dataloader

    # Reset the eval timestamp
    state.eval_timestamp = Timestamp()

    last_wct = datetime.datetime.now()

    with torch.no_grad(), model_eval_mode(state.model):
        state.set_dataloader(data_spec.dataloader, evaluator.label, subset_num_batches)
        assert state.dataloader is not None, "dataloader is set"

        engine.eval_start(state, logger)

        metrics = ensure_metrics_device_and_dtype(state, metrics)

        for metric in metrics.values():
            metric.reset()

        dataloader = state.dataloader
        dist_sampler = None
        drop_last = None
        dataset_len = None
        last_batch = False
        if isinstance(dataloader, DataLoader) and isinstance(
            dataloader.sampler, DistributedSampler
        ):
            # The distributed sampler uses `set_epoch` to set the random seed
            # Because evaluation can run on each batch, we use the batch to seed the sampler
            # so each evaluation will get a proper shuffle.
            # The epoch provided to `set_epoch` need not be sequential, so this is fine.
            dist_sampler = dataloader.sampler
            dist_sampler.set_epoch(int(state.timestamp.batch))
            drop_last = dataloader.drop_last
            # Only compute the dataset length if drop_last is False, as otherwise we don't need
            # to remove any duplicate samples.
            if drop_last == False:
                try:
                    dataset_len = len(dist_sampler.dataset)  # type: ignore
                except AttributeError:
                    warnings.warn(
                        "DistributedSampler's dataset does not have length defined. When "
                        "`drop_last=False`, metrics may be incorrect, as DistributedSampler "
                        "duplicates samples to make the dataset divisible by world size. To "
                        "fix this, provide a dataset with a length attribute to the "
                        "DistributedSampler to correctly drop duplicate samples."
                    )

        for state.batch in iter_dataloader(state, engine, logger):
            state.batch = state.device.batch_to_device(state.batch)
            if data_spec.device_transforms is not None:
                state.batch = data_spec.device_transforms(state.batch)

            # Count the batch size and num tokens before any events run
            rank_num_samples = data_spec.get_num_samples_in_batch(state.batch)
            rank_num_tokens = data_spec.get_num_tokens_in_batch(state.batch)

            # If using a distributed sampler, keep track of last_batch for metrics update
            if (
                dist_sampler is not None
                and drop_last == False
                and dataset_len is not None
            ):
                batch_num_samples_tensor = state.device.tensor_to_device(
                    torch.tensor(rank_num_samples)
                )
                dist.all_reduce(batch_num_samples_tensor, reduce_operation="SUM")
                batch_num_samples = batch_num_samples_tensor.item()
                last_batch = (
                    state.eval_timestamp.sample + batch_num_samples >= dataset_len
                )

            if state.deepspeed_enabled:
                raise ValueError(f"Deepspeed not supported")

            engine.eval_batch_start(state, logger)

            # Cache the device batch, because `self.state.batch` gets overridden in microbatching loop
            device_batch = state.batch
            # Retry until we successfully complete evaluation
            microbatches = data_spec.split_batch(
                device_batch, evaluator.device_eval_microbatch_size
            )
            for i, state.batch in enumerate(microbatches):
                last_microbatch = i == len(microbatches) - 1
                skip_metric_update = False
                # Distributed samplers pad batches to be the same size. If using a
                # distributed sampler and on last batch, remove the padding
                if (
                    dist_sampler is not None
                    and drop_last == False
                    and dataset_len is not None
                    and last_batch
                    and last_microbatch
                ):
                    padding = dist_sampler.total_size - dataset_len
                    if dist.get_global_rank() >= dist.get_world_size() - padding:
                        rank_num_samples -= 1
                        num_samples_in_microbatch = data_spec.get_num_samples_in_batch(
                            state.batch
                        )
                        # Skip updating metric if batch is only padded samples
                        if num_samples_in_microbatch == 1:
                            skip_metric_update = True
                        # Remove padded samples from batch
                        else:
                            state.batch = data_spec.split_batch(
                                state.batch, num_samples_in_microbatch - 1
                            )[0]

                engine.eval_before_forward(state, logger)

                with _get_precision_context(
                    state.precision,
                    state.precision_config,
                    state.deepspeed_enabled,
                ):
                    state.outputs = state._original_model.eval_forward(state.batch)

                engine.eval_after_forward(state, logger)

                # Skip metric update if batch is only padded samples. We do this after
                # forward as all models must run forward for FSDP.
                if skip_metric_update:
                    continue

                # Run in same precision context to avoid NaNs
                with _get_precision_context(
                    state.precision,
                    state.precision_config,
                    state.deepspeed_enabled,
                ):
                    if isinstance(state.device, DeviceMPS):
                        # torchmetrics math has numerical errors on M1 devices
                        # running the compute on CPU instead
                        outputs = state.outputs.cpu()
                    else:
                        outputs = state.outputs

                    for metric in metrics.values():
                        state._original_model.update_metric(
                            state.batch,
                            outputs,
                            metric,
                        )

            now = datetime.datetime.now()
            batch_time = now - last_wct

            total_num_samples, total_num_tokens, batch_time = (
                accumulate_time_across_ranks(
                    state,
                    num_samples=rank_num_samples,
                    num_tokens=rank_num_tokens,
                    batch_time=batch_time,
                )
            )

            state.eval_timestamp = state.eval_timestamp.to_next_batch(
                samples=total_num_samples,
                tokens=total_num_tokens,
                duration=batch_time,
            )

            last_wct = now

            engine.eval_batch_end(state, logger)

        compute_and_log_metrics(
            state, dataloader_label=evaluator.label, metrics=metrics, logger=logger
        )

        engine.eval_end(state, logger)

    state.set_dataloader(original_dataloader, original_dataloader_label)
    if original_num_batches is not None:
        state.dataloader_len = original_num_batches
