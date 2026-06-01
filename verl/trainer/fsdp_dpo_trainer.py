"""
FSDP DPO Trainer — offline Direct Preference Optimization.

Supports:
  - Standard DPO (sigmoid), IPO, Hinge loss variants
  - Full fine-tuning (separate frozen reference model)
  - LoRA fine-tuning (base model = reference, only adapter is trained)

Usage:
  torchrun --nproc_per_node=4 -m verl.trainer.fsdp_dpo_trainer --config-name dpo_trainer
"""

import logging
import os
import time
from contextlib import nullcontext
from copy import deepcopy

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import DPODataset
from verl.utils.device import auto_set_device, get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.workers.config.optimizer import build_optimizer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_DPO_LOGGING_LEVEL", "WARN"))


def _get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-sequence sum of log-probs over non-masked (label != -100) tokens.

    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len), -100 for masked positions

    Returns:
        (batch_size,) sum of log-probs per sequence
    """
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)


def _dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    loss_type: str = "sigmoid",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    Returns:
        (loss, chosen_rewards, rejected_rewards)
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    logits = chosen_rewards - rejected_rewards

    if loss_type == "sigmoid":
        if label_smoothing > 0:
            loss = (
                -F.logsigmoid(logits) * (1 - label_smoothing)
                - F.logsigmoid(-logits) * label_smoothing
            )
        else:
            loss = -F.logsigmoid(logits)
    elif loss_type == "hinge":
        loss = torch.relu(1 - logits)
    elif loss_type == "ipo":
        loss = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss type: {loss_type}. Supported: sigmoid, hinge, ipo")

    return loss.mean(), chosen_rewards.detach().mean(), rejected_rewards.detach().mean()


class FSDPDPOTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer,
        train_dataset,
        val_dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer

        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0
        self.resume_global_step = 0

        self._build_model_optimizer()
        self._init_checkpoint_manager()
        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)

        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")
        assert self.config.data.train_batch_size % dp_size == 0
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        self.train_dataloader = StatefulDataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=self.config.data.get("dataloader_num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        if val_dataset is not None:
            val_sampler = DistributedSampler(
                self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            self.val_dataloader = StatefulDataLoader(
                self.val_dataset,
                batch_size=self.config.data.micro_batch_size_per_gpu,
                sampler=val_sampler,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
            )
        else:
            self.val_dataloader = None

    def _build_model_optimizer(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "bfloat16"))

        init_context = get_init_weight_context_manager(use_meta_tensor=True)
        with init_context():
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
            )

        if self.lora:
            self.model.enable_input_require_grads()
            lora_adapter_path = self.config.model.get("lora_adapter_path")
            if lora_adapter_path is not None:
                from peft import PeftModel

                local_adapter_path = copy_to_local(lora_adapter_path)
                self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.config.model.lora_rank,
                    lora_alpha=self.config.model.lora_alpha,
                    target_modules=convert_to_regular_types(self.config.model.target_modules),
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)
            self.model = self.model.to(torch_dtype)
            # LoRA mode: reference = base model (disable_adapter)
            self.ref_model = None
        else:
            # Full fine-tune: build a separate frozen reference model
            with torch.no_grad():
                ref_init_context = get_init_weight_context_manager(use_meta_tensor=True)
                with ref_init_context():
                    self.ref_model = AutoModelForCausalLM.from_pretrained(
                        local_model_path, torch_dtype=torch_dtype, attn_implementation="flash_attention_2"
                    )
                for p in self.ref_model.parameters():
                    p.requires_grad = False

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        # FSDP wrapping
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.lora
        )

        fsdp_strategy = self.config.model.get("strategy", "fsdp")
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                param_init_fn=init_fn,
                use_orig_params=self.lora,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
            )
            if self.ref_model is not None:
                self.fsdp_ref_model = FSDP(
                    self.ref_model,
                    param_init_fn=init_fn,
                    use_orig_params=False,
                    auto_wrap_policy=get_fsdp_wrap_policy(
                        self.ref_model, config=self.config.model.fsdp_config.wrap_policy, is_lora=False
                    ),
                    device_id=get_device_id(),
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mixed_precision,
                    sync_module_states=True,
                    device_mesh=self.device_mesh,
                )
            else:
                self.fsdp_ref_model = None
        elif fsdp_strategy == "fsdp2":
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, None)
            self.fsdp_model = self.model

            if self.ref_model is not None:
                ref_state = self.ref_model.state_dict()
                apply_fsdp2(self.ref_model, fsdp_kwargs, self.config.model.fsdp_config)
                fsdp2_load_full_state_dict(self.ref_model, ref_state, self.device_mesh, None)
                self.fsdp_ref_model = self.ref_model
            else:
                self.fsdp_ref_model = None
        else:
            raise NotImplementedError(f"Unsupported FSDP strategy: {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = build_optimizer(
            [p for p in self.fsdp_model.parameters() if p.requires_grad], self.config.optim
        )

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.config.trainer.get("total_training_steps") is not None:
            self.total_steps = self.config.trainer.total_training_steps

        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)
        lr_scheduler_type = self.config.optim.get("lr_scheduler", "cosine")
        if lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif lr_scheduler_type == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {lr_scheduler_type}")

        if self.device_mesh.get_rank() == 0:
            print(f"Steps/epoch: {self.steps_per_epoch}, Total steps: {self.total_steps}")

    def _forward_logps(self, model, input_ids, attention_mask, labels):
        """Forward pass and return per-sequence log-prob sums."""
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return _get_batch_logps(outputs.logits, labels)

    def _compute_loss(self, batch, do_backward=True, n_micro_batches=1):
        """Compute DPO loss for a micro-batch."""
        chosen_input_ids = batch["chosen_input_ids"].to(self.device_name)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device_name)
        chosen_labels = batch["chosen_labels"].to(self.device_name)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device_name)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device_name)
        rejected_labels = batch["rejected_labels"].to(self.device_name)

        beta = self.config.algorithm.beta
        loss_type = self.config.algorithm.loss_type
        label_smoothing = self.config.algorithm.get("label_smoothing", 0.0)

        # Policy forward
        policy_chosen_logps = self._forward_logps(self.fsdp_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
        policy_rejected_logps = self._forward_logps(self.fsdp_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        # Reference forward
        with torch.no_grad():
            if self.lora:
                self.fsdp_model.eval()
                with self.fsdp_model.disable_adapter():
                    ref_chosen_logps = self._forward_logps(self.fsdp_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
                    ref_rejected_logps = self._forward_logps(self.fsdp_model, rejected_input_ids, rejected_attention_mask, rejected_labels)
                self.fsdp_model.train()
            else:
                ref_chosen_logps = self._forward_logps(self.fsdp_ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
                ref_rejected_logps = self._forward_logps(self.fsdp_ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        loss, chosen_rewards, rejected_rewards = _dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=beta, loss_type=loss_type, label_smoothing=label_smoothing,
        )

        loss = loss / n_micro_batches

        if do_backward:
            loss.backward()

        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss.detach(), {
            "chosen_rewards": chosen_rewards.item(),
            "rejected_rewards": rejected_rewards.item(),
            "reward_accuracy": reward_accuracy.item(),
            "reward_margin": (chosen_rewards - rejected_rewards).item(),
        }

    def training_step(self, batch):
        start_time = time.time()
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0.0
        step_metrics = {}

        for micro_batch in micro_batches:
            loss, metrics = self._compute_loss(micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()
            for k, v in metrics.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v / n_micro_batches

        fsdp_strategy = self.config.model.get("strategy", "fsdp")
        if fsdp_strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif fsdp_strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            grad_norm = torch.tensor(0.0)

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        step_loss_tensor = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss_tensor, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss_tensor)
            step_loss_tensor /= self.device_mesh.size(0)

        end_time = time.time()

        result = {
            "train/loss": step_loss_tensor.item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0.0,
            "train/time(s)": end_time - start_time,
        }
        for k, v in step_metrics.items():
            result[f"train/{k}"] = v
        return result

    def validation_step(self, batch):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, metrics = self._compute_loss(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss, metrics

    def save_checkpoint(self, step):
        from verl.utils.fs import local_mkdir_safe

        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"]) if checkpoint_config else ["model", "optimizer", "extra"]
        load_contents = checkpoint_config.get("load_contents", save_contents) if checkpoint_config else save_contents

        checkpoint_config_dict = DictConfig({
            "load_contents": load_contents,
            "save_contents": save_contents,
        })

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        checkpoint_path = None
        if resume_mode == "disable":
            return
        elif resume_mode == "auto":
            if resume_from_path and os.path.exists(resume_from_path):
                checkpoint_path = resume_from_path
            else:
                checkpoint_dir = self.config.trainer.default_local_dir
                if os.path.exists(checkpoint_dir):
                    checkpoint_path = find_latest_ckpt_path(checkpoint_dir)
        elif resume_mode == "resume_path":
            checkpoint_path = resume_from_path

        if checkpoint_path is None:
            return

        from re import search as re_search
        match = re_search(r"global_step_(\d+)", checkpoint_path)
        if match:
            self.resume_global_step = int(match.group(1))

        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        if self.device_mesh.get_rank() == 0:
            print(f"Resumed from {checkpoint_path} (step {self.resume_global_step})")

        dataloader_path = os.path.join(checkpoint_path, "data.pt")
        if os.path.exists(dataloader_path):
            state = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(state)

    def fit(self):
        rank = self.device_mesh.get_rank()

        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
                group_name=self.config.trainer.get("group_name", None),
            )

        global_step = self.resume_global_step
        start_epoch = global_step // self.steps_per_epoch if self.steps_per_epoch > 0 else 0

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch)

            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)

                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_steps
                is_save_step = self.config.trainer.save_freq > 0 and global_step % self.config.trainer.save_freq == 0
                is_valid_step = (
                    self.config.trainer.get("test_freq", -1) > 0
                    and global_step % self.config.trainer.test_freq == 0
                )

                if is_last_step or is_valid_step:
                    if self.val_dataloader is not None:
                        val_losses = []
                        val_metrics_agg = {}
                        for val_data in self.val_dataloader:
                            val_data = TensorDict(val_data, batch_size=val_data[next(iter(val_data))].shape[0]).to(
                                self.device_name
                            )
                            val_loss, val_metrics = self.validation_step(val_data)
                            val_losses.append(val_loss)
                            for k, v in val_metrics.items():
                                val_metrics_agg.setdefault(k, []).append(v)

                        if rank == 0:
                            avg_val_loss = torch.mean(torch.stack(val_losses)).item()
                            val_log = {"val/loss": avg_val_loss}
                            for k, vs in val_metrics_agg.items():
                                val_log[f"val/{k}"] = sum(vs) / len(vs)
                            tracking.log(data=val_log, step=global_step)

                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Training complete at step {global_step}")
                    return


def run_dpo(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.get("trust_remote_code", False))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_files = config.data.train_files
    if not isinstance(train_files, list):
        train_files = [train_files]

    val_files = config.data.get("val_files", None)
    if val_files and not isinstance(val_files, list):
        val_files = [val_files]

    train_dataset = DPODataset(
        parquet_files=train_files,
        tokenizer=tokenizer,
        prompt_key=config.data.get("prompt_key", "prompt"),
        chosen_key=config.data.get("chosen_key", "chosen"),
        rejected_key=config.data.get("rejected_key", "rejected"),
        max_length=config.data.get("max_length", 2048),
        max_prompt_length=config.data.get("max_prompt_length", 1024),
        max_samples=config.data.get("train_max_samples", -1),
        shuffle=True,
        seed=config.data.get("seed", 42),
    )

    val_dataset = None
    if val_files:
        val_dataset = DPODataset(
            parquet_files=val_files,
            tokenizer=tokenizer,
            prompt_key=config.data.get("prompt_key", "prompt"),
            chosen_key=config.data.get("chosen_key", "chosen"),
            rejected_key=config.data.get("rejected_key", "rejected"),
            max_length=config.data.get("max_length", 2048),
            max_prompt_length=config.data.get("max_prompt_length", 1024),
            max_samples=config.data.get("val_max_samples", -1),
            shuffle=False,
        )

    trainer = FSDPDPOTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_dpo(config)


if __name__ == "__main__":
    main()
