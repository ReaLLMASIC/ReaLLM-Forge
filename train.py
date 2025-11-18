# train.py
from contextlib import nullcontext
import csv
import json
import math
import os
import random
import pickle
import shutil
import sys
import time
from collections import deque
from datetime import datetime, timedelta

from rich.console import Group
from rich.console import Console
from rich.text import Text
from rich.live import Live


from train_variations.optimizer_variants import (
    optimizer_dictionary,
    ActRegularizedAdamW,
)
from train_variations.eta_variants import build_eta_estimator, ETAUpdate
from train_variations.loss_variants import build_loss_function
from train_variations.distillation_loss_variants import build_distillation_loss

from utils.gpu_monitoring import get_gpu_memory_info
from torch.cuda import reset_peak_memory_stats, max_memory_allocated

from utils.model_info import (
    print_summary,
    print_module_structure,
    print_model_blocks,
    print_model_tree,
)
from utils.statistic_plots import (
    initialize_statistics,
    plot_statistics,
    create_statistics,
)

from utils.model_stats import (
    compute_weight_stats,
    compute_activation_stats,
    print_model_stats_table,
)

from sample import (
    sample_with_existing_model,
    get_tokenizer_functions,
)

from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
)


# GNS Related
import utils.gns_monitoring.gns_utils as gns_utils
from utils.gns_monitoring.hook import (add_hooks_to_model, add_sogns_hooks,
                   add_exact_hooks,  gather_hook_results)

import numpy as np

# Torch
import torch
import torch.onnx
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from variations.model_variations import model_variation_dictionary

from model import GPT, GPTConfig

# Inference related imports
import tiktoken

from train_args import parse_args

class Trainer:

    def __init__(self, args, model_group, training_group, logging_group):
        self.args = args
        self.model_group = model_group
        self.training_group = training_group
        self.logging_group = logging_group

        # GNS and batch schedule
        self.gns = None
        self.grad_norm = None
        self.grad_std = None
        self.tokens_trained = 0
        self.best_tokens = 0
        self.train_limit_mode = 'iterations'
        self.limit_dataset = None
        self.limit_dataset_tokens = None
        self.limit_tokens_trained = 0
        self.limit_epochs_trained = 0.0
        self.max_epochs_target = None
        self.max_tokens_target = None
        self.eval_interval_tokens_target = None
        self.eval_interval_epochs_target = None
        self.tokens_at_last_eval = 0
        self.epochs_at_last_eval = 0.0
        self.last_eval_losses = None
        self.peak_gpu_usage = 0.0
        self.total_training_time_ms: float = 0.0   # total run-time from start of training
        self.time_remaining_ms: float= 0.0
        self.total_time_est_ms: float= 0.0
        self.evaluation_count: int = 0   # cumulative run-time (ns)
        self.evaluations_remaining: int = 0 # will be updated after the current iter is loaded
        self.formatted_completion_eta: str = "waiting for calculation"
        self.iter_latency_avg: float = 0.0  # running mean ms / iteration

        # track latest evaluation metrics for progress bar
        self.latest_top1_prob = float('nan')
        self.latest_top1_correct = float('nan')
        self.latest_target_rank = float('nan')
        self.latest_target_prob = float('nan')
        self.latest_target_left_prob = float('nan')
        self.latest_rank_95 = float('nan')
        self.latest_left_prob_95 = float('nan')
        self.latest_ln_f_cosine = float('nan')
        self.latest_ln_f_cosine_95 = float('nan')

        # store overall statistics for weights and activations
        self.latest_overall_weight_stats = {
            'stdev': 0.0,
            'kurtosis': 0.0,
            'max': 0.0,
            'min': 0.0,
            'abs_max': 0.0,
        }
        self.latest_overall_activation_stats = {
            'stdev': 0.0,
            'kurtosis': 0.0,
            'max': 0.0,
            'min': 0.0,
            'abs_max': 0.0,
        }

        # whether to show all model stats
        self.compute_model_stats = self.args.compute_model_stats

        # Where to aggregate statistics:  'cpu' (default) or 'gpu'.
        # The CLI flag is optional; fall back to CPU if it isn’t present.
        stats_dev_flag  = getattr(self.args, "model_stats_device", "cpu")
        self.stats_device = torch.device("cuda") if stats_dev_flag == "gpu" else torch.device("cpu")

        self.stats_csv_path = getattr(self.args, "print_model_stats_table", None)
        if self.stats_csv_path:
            self.stats_csv_path = os.path.expanduser(self.stats_csv_path)
            stats_dir = os.path.dirname(self.stats_csv_path)
            if stats_dir and not os.path.exists(stats_dir):
                os.makedirs(stats_dir, exist_ok=True)

        # Ensure the sample file directory exists if a path is provided
        if getattr(self.args, "sample_file", None):
            self.args.sample_file = os.path.expanduser(self.args.sample_file)
            sample_dir = os.path.dirname(self.args.sample_file)
            if sample_dir and not os.path.exists(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)

        # calculation on end time via eval cycle
        self.eval_cycle_window = deque(maxlen=self.args.eval_cycle_window)
        self.eval_cycle_latency_avg: float = 0.0
        self.eval_cycle_start_mon: bool = False

        # If using multiple datasets, track tokens trained per dataset.
        if self.args.dataset_list is not None:
            # Flatten each element (which may contain multiple dataset names) into a single list of tokens
            flattened_list = []
            for entry in self.args.dataset_list:
                flattened_list.extend(entry.split())
            self.args.dataset_list = flattened_list
            # Track tokens and epochs trained per dataset
            self.tokens_trained_dict = {dataset: 0 for dataset in self.args.dataset_list}
            self.epochs_trained_dict = {dataset: 0 for dataset in self.args.dataset_list}

            # Also, set self.args.dataset to the first dataset in the list
            self.args.dataset = self.args.dataset_list[0]
            print(self.args.dataset)

        if self.args.training_mode == 'multicontext':
            self.vocab_sizes = {}
        # init optimizer and scheduler
        self.optimizer = None
        self.scheduler = None

        # Loss function (potentially scheduled)
        self.loss_fn = build_loss_function(self.args)
        self.distillation_loss_fn = build_distillation_loss(self.args)
        self.distillation_weight = getattr(self.args, "distillation_weight", 1.0)
        self.teacher_model = None
        self.latest_distillation_loss = float('nan')
        if self.distillation_loss_fn is not None and self.args.training_mode == 'multicontext':
            raise ValueError("Knowledge distillation is not supported with multicontext training mode.")

        # Learning Rate Settings
        self.lr = self.args.learning_rate
        ## Make the decay iters equal to max_iters if not specified
        if self.args.lr_decay_match_max_iters:
            self.args.lr_decay_iters = self.args.max_iters

        self.setup()

        if self.args.sample_only:
            self.sample_and_print()

        if self.args.create_statistics:
            self.stats = initialize_statistics(self.args.n_layer, self.args.n_head)

    def setup(self):
        # Setup DDP
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend=self.args.backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            print("this is my device", self.device)
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            self.args.gradient_accumulation_steps //= self.ddp_world_size
        else:
            self.device = self.args.device
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

        self.tokens_per_iter = self.args.gradient_accumulation_steps * self.ddp_world_size * self.args.batch_size * self.args.block_size

        if self.master_process:
            os.makedirs(self.args.out_dir, exist_ok=True)

        print("seed: ", self.args.seed)
        print("seed offset: ", self.seed_offset)
        torch.manual_seed(self.args.seed + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.device_type = 'cuda' if 'cuda' in self.args.device else 'cpu'
        if self.device_type == 'cuda':
            reset_peak_memory_stats(self.device)

        self.ptdtype = {"bfloat16" : torch.bfloat16, "float16" : torch.float16, "float32" : torch.float32}[self.args.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        # Model settings
        # TODO only add if they are defined from the argparse
        self.model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions}
        self.model_args['vocab_size'] = None
        self.model_args['eval_interval'] = self.args.eval_interval

        # Training settings
        self.training_args = {action.dest: getattr(self.args, action.dest) for action in self.training_group._group_actions}
        if self.args.dataset_list is not None:
            self.model_args['lsv_dataset_num'] = len(self.args.dataset_list)
            print("self.model_args['lsv_dataset_num']")
            print(self.model_args['lsv_dataset_num'])

        if self.args.init_from == 'scratch':
            self.model_args['vocab_size'] = self.get_vocab_size_from_meta()

            # Save full configuration used for training
            config_json = {**self.model_args, **self.training_args}
            with open(self.args.out_dir + "/full_config.json", "w") as configuration_file:
                json.dump(config_json, configuration_file, indent=4)
            with open(self.args.out_dir + "/best_val_loss_and_iter.txt", 'w') as file:
                print("resetting best val loss file")


            self.load_data()
            # Initialize sampling state if using sequential or without_replacement
            if self.args.sampling_method in ["sequential", "without_replacement"]:
                if self.args.dataset_list is None:
                    available = len(self.train_data) - self.args.block_size
                    if self.args.sampling_method == "without_replacement":
                        self.indices_perm = np.random.permutation(available)
                    else:  # sequential: simply use a range
                        self.indices_perm = np.arange(available)
                    self.current_ptr = 0
                else:
                    # For each dataset in dataset_list, store a permutation and pointer.
                    self.dataset_perm = {}
                    self.dataset_ptr = {}
                    for d in self.args.dataset_list:
                        available = len(self.train_data_dict[d]) - self.args.block_size
                        if self.args.sampling_method == "without_replacement":
                            self.dataset_perm[d] = np.random.permutation(available)
                        else:
                            self.dataset_perm[d] = np.arange(available)
                        self.dataset_ptr[d] = 0

            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)
            self.model.to(self.device)

            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
            self.best_iter = 0 # for starting from scratch
            self.best_tokens = 0

            self.optimizer = self.create_optimizer()
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.args.dtype == 'float16'))
            self.scheduler = self.create_scheduler()

        elif self.args.init_from in ['resume', "prev_run"] :

            if self.args.init_from == 'resume':
                ckpt_path = os.path.join(self.args.out_dir, self.args.init_from_ckpt)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = checkpoint['iter_num']
            else:
                ckpt_path = os.path.join(self.args.prev_run_ckpt, self.args.init_from_ckpt)
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.iter_num = 0

            # we should enforce that during resume training, the identical model args are used
            checkpoint_model_args = checkpoint['model_args']
            self.model_args = checkpoint_model_args

            # support for changing select params from resume (eg. for finetuning) based on cmd-line args entered (checks if diff than defaults)
            altered_model_args = {action.dest: getattr(self.args, action.dest) for action in self.model_group._group_actions if action.default != getattr(self.args, action.dest)}
            for k in altered_model_args:
                self.model_args[k] = altered_model_args[k]

            self.load_data()
            gptconf = GPTConfig(**self.model_args)
            self.model = GPT(gptconf)

            ## TODO: Add ability here to swap WTE factors.
            state_dict = checkpoint['model']
            for k,v in list(state_dict.items()):
                if k.startswith('_orig_mod.'):
                    state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
            self.model.load_state_dict(state_dict)
            self.best_val_loss = checkpoint['best_val_loss']
            self.best_iter = checkpoint['best_iter']
            self.best_tokens = checkpoint.get('best_tokens', 0)
            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

            self.model.to(self.device)
            # Ensure optimizer and scheduler are initialized before loading state
            self.optimizer = self.create_optimizer()
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.args.dtype == 'float16'))

            self.scheduler = self.create_scheduler()

            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print("Warning: No optimizer state found in checkpoint. Using newly initialized optimizer.")

            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                print("Warning: No scheduler state found in checkpoint or scheduler is None. Using newly initialized scheduler.")


        elif self.args.init_from.startswith('gpt2'):

            assert self.args.gpt2_type in model_variation_dictionary

            self.iter_num = 0 # for starting from scratch
            self.best_val_loss = 1e9 # really big number
            self.best_iter = 0 # really big number
            self.best_tokens = 0

            variation_dict = model_variation_dictionary[self.args.gpt2_type]
            # NOTE: the hierarchy of parameters goes: 1)variation_dict >> 2)cmd-line args >> 3)GPTConfig defaults
            for k in variation_dict:
                self.model_args[k] = variation_dict[k]

            gptconf = GPTConfig(**self.model_args)
            self.model = GPT.from_pretrained(gptconf, model_type=self.args.gpt2_type)
            self.model.to(self.device)
            self.load_data()

            if self.args.lsv_focused_training:
                self.model.freeze_non_lsv_parameters()

            self.optimizer = self.create_optimizer()
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.args.dtype == 'float16'))
            self.scheduler = self.create_scheduler()

        self._initialize_teacher_if_needed()

        if self.args.block_size < self.model.config.block_size:
            self.model.crop_block_size(self.args.block_size)
            self.model_args['block_size'] = self.args.block_size

        # Add gradient monitoring
        if self.args.gns_type is not None:
            get_gns_fn = {'sogns': add_sogns_hooks, 'exact': add_exact_hooks}
            add_hooks_to_model(self.model, get_gns_fn[self.args.gns_type])
            ema_beta = self.args.gns_ema_beta
            self.gns_ema = gns_utils.EMA(beta=ema_beta)

            # Initialize GNS for later
            self.gns = None


        # Get Model Size
        self.model.num_param = self.model.get_num_params(non_embedding=False)

        # Print the model summary
        if self.args.print_model_info:
            print_summary(self.model)
            print_model_blocks(self.model)
            print_module_structure(self.model)
            print_model_tree(self.model, print_params=True)

        if self.args.compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

        self.raw_model = self.model.module if self.ddp else self.model

        if hasattr(self.loss_fn, "set_model"):
            self.loss_fn.set_model(self.raw_model)

        timestamp_prefix = time.strftime("%Y%m%d-%H%M%S")
        if self.args.timestamp:
            timestamp_prefix = self.args.timestamp

        # Tensorboard
        if self.args.tensorboard_log:
            # 1) Give the run a safe default name when the user did not supply one
            if self.args.tensorboard_run_name is None:
                self.args.tensorboard_run_name = f"{timestamp_prefix}"

            run_name = self.args.tensorboard_run_name

            # 2) Derive a *filename-safe* dataset tag (slashes ⇒ underscores)
            sanitized_dataset = self.args.dataset.replace("/", "_")

            # 3) Store a matching, safe CSV filename for later use
            if self.args.csv_log:
                self.args.csv_name = f"{sanitized_dataset}_{run_name}"
            log_subpath = os.path.join(self.args.tensorboard_log_dir, run_name)
            self.writer = SummaryWriter(log_subpath)

        # Wandb
        if self.args.wandb_log and self.master_process:
            import wandb
            self.args.csv_name = wandb_run_name
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name, config=self.args)
        self.load_tokenizer()
        self.initialize_progress_trackers()


    def _initialize_teacher_if_needed(self):
        teacher_path = getattr(self.args, "distillation_teacher_ckpt", None)
        if not teacher_path:
            if self.distillation_loss_fn is not None:
                raise ValueError(
                    "A distillation loss was selected but no teacher checkpoint was provided via --distillation_teacher_ckpt."
                )
            return

        if self.distillation_loss_fn is None:
            raise ValueError(
                "A teacher checkpoint was supplied without selecting a distillation loss variant. Use --distillation_loss."
            )

        expanded = os.path.expanduser(teacher_path)
        if not os.path.exists(expanded):
            candidate = os.path.join(self.args.out_dir, teacher_path)
            if os.path.exists(candidate):
                expanded = candidate

        checkpoint = torch.load(expanded, map_location=self.device)
        teacher_args = checkpoint.get('model_args')
        if teacher_args is None:
            raise ValueError("Teacher checkpoint does not contain 'model_args'.")

        teacher_config = GPTConfig(**teacher_args)
        teacher_model = GPT(teacher_config)

        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if key.startswith('_orig_mod.'):
                state_dict[key[len('_orig_mod.'):]] = state_dict.pop(key)

        teacher_model.load_state_dict(state_dict)
        teacher_model.to(self.device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)

        self.teacher_model = teacher_model
        if self.master_process:
            print(f"Loaded teacher checkpoint from {expanded}")


    def create_optimizer(self):
        optimizer_key = self.args.optimizer

        if optimizer_key == "muon":
            named = list(self.model.named_parameters())
            exclude = ("embed", "wte", "wpe", "lm_head")
            hidden = [p for n, p in named if p.ndim >= 2 and all(e not in n for e in exclude)]
            other = [p for n, p in named if not (p.ndim >= 2 and all(e not in n for e in exclude))]
            param_groups = [
                {"params": other, "use_muon": False},
                {"params": hidden, "use_muon": True},
            ]
        else:
            param_groups = [
                {"params": self.model.parameters(), "lr": self.args.learning_rate}
            ]

        try:
            optimizer_builder = optimizer_dictionary[optimizer_key]
        except KeyError:
            raise ValueError(f"Unknown optimizer '{optimizer_key}'. "
                             f"Available: {list(optimizer_dictionary)}")

        optimizer = optimizer_builder(param_groups, self.args)

        return optimizer

    def create_scheduler(self):
        if self.args.lr_scheduler == "none":
            return None
        elif self.args.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.cosine_t_max, eta_min=self.args.cosine_eta_min)
        elif self.args.lr_scheduler == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.exponential_gamma)
        elif self.args.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_lr_size, gamma=self.args.step_lr_gamma)
        elif self.args.lr_scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode=self.args.plateau_mode, factor=self.args.plateau_factor, patience=self.args.plateau_patience)
        else:
            raise ValueError(f"Unknown scheduler: {self.args.lr_scheduler}")

    def load_tokenizer(self):
        if self.args.dataset_list is not None and self.args.multidataset_wte:
            self.encode_dict = {}
            self.decode_dict = {}
            for dataset in self.args.dataset_list:
                meta_path = os.path.join('data', dataset, 'meta.pkl')
                if not os.path.exists(meta_path):
                    sys.exit(f"Error: meta.pkl not found for {dataset}")
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                encode, decode = get_tokenizer_functions(meta)
                self.encode_dict[dataset] = encode
                self.decode_dict[dataset] = decode
            self.encode = self.encode_dict[self.args.dataset_list[0]]
            self.decode = self.decode_dict[self.args.dataset_list[0]]
        else:
            meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)

                self.encode, self.decode = get_tokenizer_functions(meta)

                if 'tokenizer' in meta:
                    if meta['tokenizer'] == 'sentencepiece':
                        self.separator_token = "▁"
                    print(f"Using {meta['tokenizer']} tokenizer")
                else:
                    print("Using default character-level tokenizer")

                if 'stoi' in meta:
                    self.stoi = meta['stoi']
                if 'itos' in meta:
                    self.itos = meta['itos']
            else:
                sys.exit("Error: meta.pkl not found")

    def initialize_progress_trackers(self):
        # Determine which dataset governs epoch/token based schedules.
        self.limit_dataset_tokens = None
        if isinstance(self.dataset_size_tokens, dict):
            if self.args.training_mode == 'multidataset' and self.args.dataset_list:
                self.limit_dataset = self.args.dataset_list[0]
            elif self.args.training_mode == 'multicontext' and self.args.multicontext_datasets:
                self.limit_dataset = self.args.multicontext_datasets[0]
            else:
                self.limit_dataset = next(iter(self.dataset_size_tokens), None)
            if self.limit_dataset is not None:
                self.limit_dataset_tokens = self.dataset_size_tokens.get(self.limit_dataset)
        else:
            self.limit_dataset_tokens = self.dataset_size_tokens

        # Reset derived tracking state.
        self.limit_tokens_trained = self.get_limit_tokens_trained()
        if self.limit_dataset_tokens:
            self.limit_epochs_trained = self.limit_tokens_trained / self.limit_dataset_tokens
        else:
            self.limit_epochs_trained = 0.0
        self.tokens_at_last_eval = self.limit_tokens_trained
        self.epochs_at_last_eval = self.limit_epochs_trained

        # Establish schedule overrides for epoch/token limited training.
        self.train_limit_mode = 'iterations'
        self.max_epochs_target = None
        self.max_tokens_target = None
        self.eval_interval_tokens_target = None
        self.eval_interval_epochs_target = None

        if self.args.max_epochs is not None:
            if self.limit_dataset_tokens is None:
                raise ValueError('Unable to determine dataset size for --max_epochs scheduling.')
            self.train_limit_mode = 'epochs'
            self.max_epochs_target = float(self.args.max_epochs)
            self.max_tokens_target = int(math.ceil(self.args.max_epochs * self.limit_dataset_tokens))
            interval_epochs = self.args.eval_interval_epochs if self.args.eval_interval_epochs is not None else 1.0
            self.eval_interval_epochs_target = float(interval_epochs)
            interval_tokens = max(1, int(math.ceil(interval_epochs * self.limit_dataset_tokens)))
            self.eval_interval_tokens_target = interval_tokens
        elif self.args.max_tokens is not None:
            self.train_limit_mode = 'tokens'
            self.max_tokens_target = int(self.args.max_tokens)
            interval_tokens = self.args.eval_interval_tokens
            if interval_tokens is None:
                interval_tokens = self.args.eval_interval * self.tokens_per_iter
            interval_tokens = max(1, int(math.ceil(interval_tokens)))
            self.eval_interval_tokens_target = interval_tokens

        if self.train_limit_mode in ('epochs', 'tokens'):
            self.args.max_iters = max(1, int(math.ceil(self.max_tokens_target / self.tokens_per_iter)))
            if self.eval_interval_tokens_target is not None:
                self.args.eval_interval = max(1, int(math.ceil(self.eval_interval_tokens_target / self.tokens_per_iter)))
            if getattr(self.args, 'lr_decay_match_max_iters', False):
                self.args.lr_decay_iters = self.args.max_iters

        # Keep auxiliary structures in sync with derived arguments.
        self.model_args['eval_interval'] = self.args.eval_interval
        if hasattr(self, 'training_args'):
            self.training_args['max_iters'] = self.args.max_iters
            self.training_args['eval_interval'] = self.args.eval_interval

    def get_limit_tokens_trained(self):
        if self.args.dataset_list is not None and hasattr(self, 'tokens_trained_dict'):
            target = self.limit_dataset or self.args.dataset_list[0]
            return self.tokens_trained_dict.get(target, 0)
        return self.tokens_trained

    def update_limit_tracking(self):
        if self.train_limit_mode not in ('epochs', 'tokens'):
            return
        self.limit_tokens_trained = self.get_limit_tokens_trained()
        if self.limit_dataset_tokens:
            self.limit_epochs_trained = self.limit_tokens_trained / self.limit_dataset_tokens
        else:
            self.limit_epochs_trained = 0.0

    def should_eval_for_tokens(self):
        if self.eval_interval_tokens_target is None:
            return False
        current_tokens = self.get_limit_tokens_trained()
        return (current_tokens - self.tokens_at_last_eval) >= self.eval_interval_tokens_target

    def should_eval_for_epochs(self):
        if self.eval_interval_epochs_target is None:
            return False
        if self.limit_dataset_tokens is None:
            return False
        current_epochs = self.limit_epochs_trained
        return (current_epochs - self.epochs_at_last_eval) >= self.eval_interval_epochs_target

    def should_eval_for_limits(self):
        return self.should_eval_for_tokens() or self.should_eval_for_epochs()

    @torch.no_grad()
    def sample_and_print(self):
        self.console.rule("[bold green]Inference Samples[/bold green]")
        # Do one iteration per lsv, default to one with no lsv
        sample_iterations = 1

        self.model.eval()

        if self.args.dataset_list is not None:
            sample_iterations = len(self.args.dataset_list)

        for i in range(sample_iterations):
            if self.args.use_lsv:
                self.model.set_lsv_index(i)
                print(f"lsv index {i}")

            if hasattr(self, 'encode_dict'):
                encode_fn = self.encode_dict[self.args.dataset_list[i]]
                decode_fn = self.decode_dict[self.args.dataset_list[i]]
            else:
                encode_fn = self.encode
                decode_fn = self.decode

            start_ids = torch.tensor(encode_fn(self.args.sample_start_tokens), dtype=torch.long, device=self.device)[None, ...]

            with torch.no_grad():
                sample_with_existing_model(
                    model=self.model,
                    start_ids=start_ids,
                    start_tokens=self.args.sample_start_tokens,
                    decode=decode_fn,
                    device=self.device,
                    out_dir=self.args.out_dir,
                    max_new_tokens=self.args.max_sample_tokens,
                    temperature=self.args.temperature,
                    top_k=self.args.top_k,
                    colorize_output=self.args.colorize_output,
                    colorize_mode=self.args.colorize_mode,
                    token_boundary=(self.args.token_boundary or None),
                    show_heatmaps=self.args.show_heatmaps,
                    sample_file=self.args.sample_file,
                    num_samples=self.args.num_samples,
                    iter_num=self.iter_num,
                    best_val_loss=self.best_val_loss,
                    run_name=self.args.tensorboard_run_name,
                    args=self.args,
                    writer=self.writer if self.args.tensorboard_log else None,
                    dataset_idx=i if hasattr(self, 'encode_dict') else None,
                    console=self.console,
                )

        # After sampling from the model, optionally run simple dataset benchmarks
        if self.args.dataset_benchmarks and self.args.max_sample_tokens:
            self.run_dataset_benchmarks()

        self.model.train()
        self.console.rule("[bold green]End Samples[/bold green]")
        self.console.print("\n"*8)

    def get_vocab_size_from_meta(self):
        # Data loader
        meta_path = os.path.join('data', self.args.dataset, 'meta.pkl')
        # Save a copy of meta.pkl tokenization into the output folder
        self.copy_file_to_directory(meta_path, self.args.out_dir)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
                if 'vocab_size' in meta:
                    return meta['vocab_size']
                else:
                    sys.exit(f"Error: 'vocab_size' key not found in {meta_path}")
        else:
            sys.exit(f"Error: File not found - {meta_path}")

    def copy_file_to_directory(self, src_file, dest_dir):
        try:
            # Ensure the destination directory exists
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Copy the file
            shutil.copy(src_file, dest_dir)
            print(f"File {src_file} copied to {dest_dir}")
        except Exception as e:
            print(f"Error copying file: {e}")

    def run_dataset_benchmarks(self):
        """Sample a chunk of dataset text and print simple metrics."""
        try:
            if hasattr(self, "train_data"):
                data = self.train_data
                decode_fn = self.decode
            elif hasattr(self, "train_data_dict") and self.args.dataset_list:
                first_ds = self.args.dataset_list[0]
                data = self.train_data_dict[first_ds]
                decode_fn = self.decode_dict[first_ds] if hasattr(self, 'decode_dict') else self.decode
            else:
                return

            if len(data) < self.args.max_sample_tokens:
                return

            start = random.randint(0, len(data) - self.args.max_sample_tokens)
            ids = data[start : start + self.args.max_sample_tokens].astype(int)
            text = decode_fn(ids.tolist())
            from benchmarks import run_all

            metrics = run_all(text)
            print("Dataset sample metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.3f}")
            if self.args.tensorboard_log and self.writer is not None:
                for mk, mv in metrics.items():
                    self.writer.add_scalar(f"dataset_benchmarks/{mk}", mv, self.iter_num)
        except Exception as e:
            print(f"Error running dataset benchmarks: {e}")

    def load_data(self):

        if self.args.training_mode == 'multicontext':
            # Expecting --multicontext_datasets to be provided.
            if self.args.multicontext_datasets is None:
                sys.exit("Error: When training_mode is 'multicontext', please provide --multicontext_datasets.")
            self.train_data_dict = {}
            self.val_data_dict = {}
            for dataset in self.args.multicontext_datasets:
                meta_path = os.path.join('data', dataset, 'meta.pkl')
                if not os.path.exists(meta_path):
                    sys.exit(f"Error: Meta file not found at {meta_path}")
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    vocab_size = meta.get('vocab_size', None)
                    print(vocab_size, dataset)
                    self.vocab_sizes[dataset] = meta['vocab_size']
                # Here we use np.uint16 for most datasets:
                self.train_data_dict[dataset] = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=np.uint16, mode='r')
                self.val_data_dict[dataset]   = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=np.uint16, mode='r')

            # Also store total token counts per dataset.
            self.dataset_size_tokens = {d: len(self.train_data_dict[d]) for d in self.args.multicontext_datasets}
            # tell the model we are in "multicontext" mode and pass
            #         the (ordered) list of vocab sizes it needs.
            self.model_args['multicontext'] = True
            self.model_args['vocab_sizes'] = [
                self.vocab_sizes[d] for d in self.args.multicontext_datasets
            ]

            # Let the first of the vocab sizes be used for calculation of btc
            self.model_args['vocab_size'] = self.model_args['vocab_sizes'][0]
        if self.args.training_mode == 'multidataset':
            self.train_data_dict = {}
            self.val_data_dict = {}
            self.vocab_sizes = []

            for dataset in self.args.dataset_list:
                train_data = None
                val_data = None
                meta_path = os.path.join('data', dataset, 'meta.pkl')
                if not os.path.exists(meta_path):
                    sys.exit(f"Error: Meta file not found at {meta_path}")

                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    vocab_size = meta.get('vocab_size', None)
                    if vocab_size:
                        self.vocab_sizes.append(vocab_size)

                # Load train and val data for each dataset
                dtype = np.uint16 if vocab_size != 100277 else np.uint32
                train_data = np.memmap(os.path.join('data', dataset, 'train.bin'), dtype=dtype, mode='r')
                val_data = np.memmap(os.path.join('data', dataset, 'val.bin'), dtype=dtype, mode='r')

                # Store in dictionaries
                self.train_data_dict[dataset] = train_data
                self.val_data_dict[dataset] = val_data

            self.dataset_size_tokens = {d: len(self.train_data_dict[d]) for d in self.args.dataset_list}

            if self.args.multidataset_wte:
                self.model_args['multidataset_wte'] = True
                self.model_args['vocab_sizes'] = self.vocab_sizes
                self.model_args['vocab_size'] = self.vocab_sizes[0]
            else:
                self.model_args['vocab_size'] = max(self.vocab_sizes)
        else:

            if self.model_args['vocab_size'] is None:
                sys.exit("Error: no vocab size specified")
            elif self.model_args['vocab_size'] == 100277:
                # cl100k_base, vocab size 100277, requires np.uint32
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint32, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint32, mode='r')
            else:
                # all other tokenations so far require only np.uint16
                self.train_data = np.memmap(os.path.join('data', self.args.dataset, 'train.bin'), dtype=np.uint16, mode='r')
                self.val_data = np.memmap(os.path.join('data', self.args.dataset, 'val.bin'), dtype=np.uint16, mode='r')
            # Store total token count for the single dataset.
            self.dataset_size_tokens = len(self.train_data)


    def get_batch(self, split, target_dataset=None):
        dataset = None
        data = None
        def interpolate_probs(initial_probs, final_probs, method, step_ratio):
            if method == 'linear':
                return initial_probs + step_ratio * (final_probs - initial_probs)
            elif method == 'cosine':
                return initial_probs + 0.5 * (1 - np.cos(np.pi * step_ratio)) * (final_probs - initial_probs)
            elif method == 'exponential':
                return initial_probs * (final_probs / initial_probs) ** step_ratio
            else:
                raise ValueError(f"Unknown transition method: {method}")

        def get_transitioned_probs():
            initial_probs = np.array(self.args.dataset_sampling_probs)
            if self.args.dataset_sampling_probs_final:
                step_ratio = self.iter_num / self.args.max_iters
                final_probs = np.array(self.args.dataset_sampling_probs_final)
                return interpolate_probs(initial_probs, final_probs, self.args.dataset_sampling_probs_transition_method, step_ratio)
            return initial_probs

        if self.args.training_mode == 'multicontext':
            x_dict = {}
            y_dict = {}
            ix = None
            # For each context/dataset, grab a batch
            for dataset_name in self.args.multicontext_datasets:
                data = (self.train_data_dict[dataset_name]
                        if split == 'train' else self.val_data_dict[dataset_name])
                if ix is None:
                    ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
                # pick random offset
                x = torch.stack([
                    torch.from_numpy(data[i : i+self.args.block_size].astype(np.int64))
                    for i in ix
                    ])
                y = torch.stack([
                    torch.from_numpy(data[i+1 : i+1+self.args.block_size].astype(np.int64))
                    for i in ix
                    ])
                # Move to device
                if self.device_type == 'cuda':
                    x = x.pin_memory().to(self.device, non_blocking=True)
                    y = y.pin_memory().to(self.device, non_blocking=True)
                else:
                    x, y = x.to(self.device), y.to(self.device)

                x_dict[dataset_name] = x
                y_dict[dataset_name] = y

            return x_dict, y_dict, list(self.args.multicontext_datasets)

        elif self.args.training_mode == "multidataset":
            # If multi-dataset sampling is enabled, pick a dataset using sampling probabilities
            if target_dataset:
                dataset = target_dataset
                data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]
            elif self.args.dataset_interleaving:
                # print("using interleaving")
                if self.args.dataset_sampling_probs is not None:
                    # TODO: Move this section into README
                    # sampling proportions in this case
                    # allows for interleaving datasets
                    # Option 1) specific complex order
                    # a b a a b
                    # 1 1 1 1 1
                    # output: a b a a b
                    # Option 2) specific ratio shorthand
                    # a b c
                    # 1 3 2
                    # output: a b b b c c
                    # Option 3) specific ratio with random shuffle
                    # a b c
                    # 1 2 3
                    # possible random output: c a b c b c

                    # Init if does not exist
                    if not hasattr(self, 'remaining_datasets'):
                        self.remaining_datasets = []
                        # print("init")

                    # Reset if zero remaining
                    if len(self.remaining_datasets) == 0:
                        self.remaining_datasets = [x for x, count in zip(self.args.dataset_list, self.args.dataset_sampling_probs) for _ in range(int(count))]

                        # shuffle
                        if self.args.dataset_interleaving_shuffle:
                            random.shuffle(self.remaining_datasets)
                        # print("reset", self.remaining_datasets)

                    # pop from front of stack
                    dataset = self.remaining_datasets.pop(0)
                    # print("dataset", dataset, "remaining", self.remaining_datasets)
                else:
                    # If proportions and order not specified, then do 1:1 interleaving
                    num_datasets = len(self.args.dataset_list)
                    dataset_index = self.iter_num % num_datasets
                    dataset = self.args.dataset_list[dataset_index]

                data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]
            else:
                # print("using probabilities")
                if self.args.dataset_sampling_probs:
                    # Sample dataset based on probabilities
                    dataset = np.random.choice(self.args.dataset_list, p=get_transitioned_probs() / np.sum(get_transitioned_probs()))
                else:
                    # Default to uniform sampling if probabilities are not provided
                    dataset = np.random.choice(self.args.dataset_list)
                # print(dataset)
                data = self.train_data_dict[dataset] if split == 'train' else self.val_data_dict[dataset]

            if self.args.use_lsv:
                self.model.set_lsv_index(self.args.dataset_list.index(dataset))


            # set learning rate
            if self.args.dataset_sampling_learning_rate:
                dataset_index = self.args.dataset_list.index(dataset)
                self.args.learning_rate = self.args.dataset_sampling_learning_rate[dataset_index]

        else:
            # Else use the 'dataset' arg by default for backwards compatibility
            dataset = self.args.dataset
            data = self.train_data if split == 'train' else self.val_data

        # Adaptive GNS settings
        if (self.gns is not None) and (self.args.gns_target is not None):
            if self.gns < self.args.gns_target:
                if self.args.batch_size < self.args.gns_max_batch:
                    self.args.batch_size = math.ceil(self.args.batch_size * (1.0 + self.args.gns_batch_pct))
            if self.gns > self.args.gns_target:
                self.args.batch_size = math.ceil(self.args.batch_size * (1.0 - self.args.gns_batch_pct))

        # Generate random indices for the batch
        ix = torch.randint(len(data) - self.args.block_size, (self.args.batch_size,))
        available = len(data) - self.args.block_size
        if self.args.sampling_method == "random":
            ix = torch.randint(available, (self.args.batch_size,))
        elif self.args.sampling_method == "sequential":
            # Use the sequential indices from self.indices_perm (or per dataset)
            if self.args.dataset_list is None:
                if self.current_ptr + self.args.batch_size > available:
                    self.current_ptr = 0
                selected_indices = self.indices_perm[self.current_ptr: self.current_ptr + self.args.batch_size]
                self.current_ptr += self.args.batch_size
            else:
                d = target_dataset if target_dataset is not None else self.args.dataset
                if self.dataset_ptr[d] + self.args.batch_size > available:
                    self.dataset_ptr[d] = 0
                selected_indices = self.dataset_perm[d][self.dataset_ptr[d]: self.dataset_ptr[d] + self.args.batch_size]
                self.dataset_ptr[d] += self.args.batch_size
            ix = torch.tensor(selected_indices)
        elif self.args.sampling_method == "without_replacement":
            # Similar to sequential but with a shuffled permutation that is reshuffled when exhausted.
            if self.args.dataset_list is None:
                if self.current_ptr + self.args.batch_size > available:
                    self.indices_perm = np.random.permutation(available)
                    self.current_ptr = 0
                selected_indices = self.indices_perm[self.current_ptr: self.current_ptr + self.args.batch_size]
                self.current_ptr += self.args.batch_size
            else:
                d = target_dataset if target_dataset is not None else self.args.dataset
                if self.dataset_ptr[d] + self.args.batch_size > available:
                    self.dataset_perm[d] = np.random.permutation(available)
                    self.dataset_ptr[d] = 0
                selected_indices = self.dataset_perm[d][self.dataset_ptr[d]: self.dataset_ptr[d] + self.args.batch_size]
                self.dataset_ptr[d] += self.args.batch_size
            ix = torch.tensor(selected_indices)
        else:
            # Default to random sampling if unknown method
            ix = torch.randint(available, (self.args.batch_size,))


        # Get training and targets
        x = torch.stack([torch.from_numpy((data[i:i+self.args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.args.block_size]).astype(np.int64)) for i in ix])

        # Send to appropriate device
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y, dataset

    @torch.no_grad()
    def estimate_loss(self):
        out = {'datasets':{}}

        self.model.eval()
        # If multi-dataset sampling is enabled, we calculate loss per dataset
        if self.args.dataset_list:
            for dataset in self.args.dataset_list:
                print(f"Calculating loss for dataset: {dataset}")
                dataset_losses = {'train': torch.zeros(self.args.eval_iters), 'val': torch.zeros(self.args.eval_iters)}
                top1_probs, top1_corrects, target_ranks, target_probs, target_left_probs, left_inclusive_probs = [], [], [], [], [], []
                ln_f_cosines = []
                for split in ['train', 'val']:
                    for k in range(self.args.eval_iters):
                        X, Y, test_dataset = self.get_batch(split, target_dataset=dataset)
                        ln_f_out: list[torch.Tensor] = []
                        handle = self.model.transformer.ln_f.register_forward_hook(
                            lambda _m, _i, o: ln_f_out.append(o.detach())
                        )
                        with self.ctx:
                            idx = self.args.dataset_list.index(dataset)
                            logits, loss = self.model(
                                X,
                                Y,
                                iter_num=self.iter_num,
                                dataset_idx=idx if self.args.multidataset_wte else None,
                                loss_fn=self.loss_fn,
                            )
                        handle.remove()
                        dataset_losses[split][k] = loss.item()
                        if split == 'val':
                            probs = F.softmax(logits, dim=-1)
                            top1_prob, top1_idx = probs.max(dim=-1)
                            top1_probs.append(top1_prob)
                            top1_corrects.append((top1_idx == Y).float())
                            target_logits = logits.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
                            ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
                            target_ranks.append(ranks.float())
                            target_prob = probs.gather(-1, Y.unsqueeze(-1)).squeeze(-1).float()
                            target_probs.append(target_prob)
                            left_prob = (probs * (probs > target_prob.unsqueeze(-1))).sum(dim=-1).float()
                            target_left_probs.append(left_prob)
                            left_inclusive_probs.append(left_prob + target_prob)
                            lm_head = (
                                self.model.transformer[f'lm_head_{idx}']
                                if self.args.multidataset_wte
                                else self.model.lm_head
                            )
                            target_vecs = lm_head.weight[Y]
                            cos = F.cosine_similarity(
                                ln_f_out[0].float(), target_vecs.float(), dim=-1
                            )
                            ln_f_cosines.append(cos)
                out['datasets'][dataset] = {
                        'train': dataset_losses['train'].mean(),
                        'train_std': dataset_losses['train'].std(),
                        'val': dataset_losses['val'].mean(),
                        'val_std': dataset_losses['val'].std(),
                        'top1_prob': torch.cat(top1_probs).mean() if top1_probs else torch.tensor(float('nan')),
                        'top1_correct': torch.cat(top1_corrects).mean() if top1_corrects else torch.tensor(float('nan')),
                        'target_rank': torch.cat(target_ranks).mean() if target_ranks else torch.tensor(float('nan')),
                        'target_left_prob': torch.cat(target_left_probs).mean() if target_left_probs else torch.tensor(float('nan')),
                        'target_prob': torch.cat(target_probs).mean() if target_probs else torch.tensor(float('nan')),
                        'target_rank_95': torch.quantile(torch.cat(target_ranks), 0.95) if target_ranks else torch.tensor(float('nan')),
                        'left_prob_95': torch.quantile(torch.cat(left_inclusive_probs).float(), 0.95) if left_inclusive_probs else torch.tensor(float('nan')),
                        'ln_f_cosine': torch.cat(ln_f_cosines).mean() if ln_f_cosines else torch.tensor(float('nan')),
                        'ln_f_cosine_95': torch.quantile(torch.cat(ln_f_cosines), 0.05) if ln_f_cosines else torch.tensor(float('nan')),
                        }
            out['val'] = out['datasets'][self.args.dataset]['val']
            out['val_std'] = out['datasets'][self.args.dataset]['val_std']
            out['train'] = out['datasets'][self.args.dataset]['train']
            out['train_std'] = out['datasets'][self.args.dataset]['train_std']
            out['top1_prob'] = out['datasets'][self.args.dataset]['top1_prob']
            out['top1_correct'] = out['datasets'][self.args.dataset]['top1_correct']
            out['target_rank'] = out['datasets'][self.args.dataset]['target_rank']
            out['target_left_prob'] = out['datasets'][self.args.dataset]['target_left_prob']
            out['target_prob'] = out['datasets'][self.args.dataset]['target_prob']
            out['target_rank_95'] = out['datasets'][self.args.dataset]['target_rank_95']
            out['left_prob_95'] = out['datasets'][self.args.dataset]['left_prob_95']
            out['ln_f_cosine'] = out['datasets'][self.args.dataset]['ln_f_cosine']
            out['ln_f_cosine_95'] = out['datasets'][self.args.dataset]['ln_f_cosine_95']
        elif self.args.training_mode == "multicontext":
            for i, dataset in enumerate(self.args.multicontext_datasets):
                out['datasets'][dataset] = {}
            # multicontext training
            for split in ['train', 'val']:
                losses = {}
                means = {}
                std_devs = {}
                mean_avg = 0.0
                loss_std = 0.0
                dataset_list = None
                for i, dataset in enumerate(self.args.multicontext_datasets):
                    losses[f"{i}"] = torch.zeros(self.args.eval_iters)
                    means[f"{i}"] = 0.0

                for k in range(self.args.eval_iters):
                    x_dict, y_dict, dataset_list = self.get_batch(split)

                    with self.ctx:
                        logits, loss_list = self.model(
                            None,
                            token_dict=x_dict,
                            target_dict=y_dict,
                            iter_num=self.iter_num,
                            loss_fn=self.loss_fn,
                        )
                    for i in range(len(self.args.multicontext_datasets)):
                        losses[f"{i}"][k] = loss_list[i]

                for i, dataset in enumerate(self.args.multicontext_datasets):
                    means[f"{i}"] = losses[f"{i}"].mean()
                    std_devs[f"{i}"]  = losses[f"{i}"].std()

                    mean_avg += means[f"{i}"]
                    loss_std += std_devs[f"{i}"]

                for i, dataset in enumerate(self.args.multicontext_datasets):
                    out['datasets'][dataset][split] = means[f"{i}"]
                    out['datasets'][dataset][f"{split}_std"] = std_devs[f"{i}"]

                # general train and val losses, as well as std dev
                out[split] = np.array(mean_avg / len(self.args.multicontext_datasets))
                out[split + "_std"] = np.array(loss_std / len(self.args.multicontext_datasets))
        else:
            # Default behavior for a single dataset
            for split in ['train', 'val']:
                losses = torch.zeros(self.args.eval_iters)
                top1_probs, top1_corrects, target_ranks, target_probs, target_left_probs, left_inclusive_probs = [], [], [], [], [], []
                ln_f_cosines = []
                for k in range(self.args.eval_iters):
                    X, Y, _ = self.get_batch(split)
                    ln_f_out: list[torch.Tensor] = []
                    handle = self.model.transformer.ln_f.register_forward_hook(
                        lambda _m, _i, o: ln_f_out.append(o.detach())
                    )
                    with self.ctx:
                        logits, loss = self.model(
                            X,
                            Y,
                            iter_num=self.iter_num,
                            dataset_idx=0 if self.args.multidataset_wte else None,
                            loss_fn=self.loss_fn,
                        )
                    handle.remove()
                    losses[k] = loss.item()
                    if split == 'val':
                        probs = F.softmax(logits, dim=-1)
                        top1_prob, top1_idx = probs.max(dim=-1)
                        top1_probs.append(top1_prob)
                        top1_corrects.append((top1_idx == Y).float())
                        target_logits = logits.gather(-1, Y.unsqueeze(-1)).squeeze(-1)
                        ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1
                        target_ranks.append(ranks.float())
                        target_prob = probs.gather(-1, Y.unsqueeze(-1)).squeeze(-1).float()
                        target_probs.append(target_prob)
                        left_prob = (probs * (probs > target_prob.unsqueeze(-1))).sum(dim=-1).float()
                        target_left_probs.append(left_prob)
                        left_inclusive_probs.append(left_prob + target_prob)
                        lm_head = (
                            self.model.transformer['lm_head_0']
                            if self.args.multidataset_wte
                            else self.model.lm_head
                        )
                        target_vecs = lm_head.weight[Y]
                        cos = F.cosine_similarity(
                            ln_f_out[0].float(), target_vecs.float(), dim=-1
                        )
                        ln_f_cosines.append(cos)
                out[split] = losses.mean()
                out[split + "_std"] = losses.std()
                if split == 'val':
                    out['top1_prob'] = torch.cat(top1_probs).mean() if top1_probs else torch.tensor(float('nan'))
                    out['top1_correct'] = torch.cat(top1_corrects).mean() if top1_corrects else torch.tensor(float('nan'))
                    out['target_rank'] = torch.cat(target_ranks).mean() if target_ranks else torch.tensor(float('nan'))
                    out['target_left_prob'] = torch.cat(target_left_probs).mean() if target_left_probs else torch.tensor(float('nan'))
                    out['target_prob'] = torch.cat(target_probs).mean() if target_probs else torch.tensor(float('nan'))
                    out['target_rank_95'] = torch.quantile(torch.cat(target_ranks), 0.95) if target_ranks else torch.tensor(float('nan'))
                    out['left_prob_95'] = torch.quantile(torch.cat(left_inclusive_probs).float(), 0.95) if left_inclusive_probs else torch.tensor(float('nan'))
                    out['ln_f_cosine'] = torch.cat(ln_f_cosines).mean() if ln_f_cosines else torch.tensor(float('nan'))
                    out['ln_f_cosine_95'] = torch.quantile(torch.cat(ln_f_cosines), 0.05) if ln_f_cosines else torch.tensor(float('nan'))

        # compute statistics from a single validation batch
        if self.compute_model_stats:
            X_stat, Y_stat, _ = self.get_batch('val')
            # ── Run heavy ops on the selected device (GPU keeps host‑RAM flat) ──
            act_stats,  overall_act  = compute_activation_stats(
                    self.model, X_stat, Y_stat, self.iter_num, device=self.stats_device
                    )
            weight_stats, overall_wt = compute_weight_stats(
                    self.model, device=self.stats_device
                    )

            self.latest_overall_weight_stats     = overall_wt
            self.latest_overall_activation_stats = overall_act

            print_model_stats_table(weight_stats, act_stats, csv_path=self.stats_csv_path, console=self.console)
        else:
            act_stats  = {}   # keep API intact
            weight_stats = {}

        if self.args.tensorboard_log and self.compute_model_stats:
            self.writer.add_scalars(
                    "model_stats",
                    {
                        "weight_stdev": overall_wt['stdev'],
                        "weight_kurtosis": overall_wt['kurtosis'],
                        "weight_max": overall_wt['max'],
                        "weight_min": overall_wt['min'],
                        "weight_abs_max": overall_wt['abs_max'],
                        "activation_stdev": overall_act['stdev'],
                        "activation_kurtosis": overall_act['kurtosis'],
                        "activation_max": overall_act['max'],
                        "activation_min": overall_act['min'],
                        "activation_abs_max": overall_act['abs_max'],
                        },
                    self.iter_num,
                    )

            # Log per-tensor stats grouped by statistic
            for stat_key in ["stdev", "kurtosis", "max", "min", "abs_max"]:
                if weight_stats:
                    self.writer.add_scalars(
                            f"weights/{stat_key}",
                            {n: s[stat_key] for n, s in weight_stats.items()},
                            self.iter_num,
                            )
                if act_stats:
                    self.writer.add_scalars(
                            f"activations/{stat_key}",
                            {n: s[stat_key] for n, s in act_stats.items()},
                            self.iter_num,
                            )

        self.model.train()
        return out


    def get_lr(self, it):
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        if it < self.args.warmup_iters:
            return self.args.learning_rate * it / self.args.warmup_iters
        if it > self.args.lr_decay_iters:
            return self.args.min_lr
        decay_ratio = (it - self.args.warmup_iters) / (self.args.lr_decay_iters - self.args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.args.min_lr + coeff * (self.args.learning_rate - self.args.min_lr)


    @torch.no_grad()
    def get_gradient_stats(self):
        """
        Calculates and returns the gradient standard deviation, norm, and mean for a PyTorch model.

        Args:
            model: The PyTorch model.

        Returns:
            A dictionary containing the gradient standard deviation, norm, and mean.  Returns None if no gradients are available.
        """

        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:  # Check if gradients exist for the parameter
                gradients.append(param.grad.view(-1)) # Flatten and append the gradients
            # Handle cases where some parameters might not have gradients (e.g. frozen layers)
        if not gradients:
            return None # No gradients found

        all_gradients = torch.cat(gradients) # Concatenate all gradients into a single tensor

        self.grad_std = torch.std(all_gradients).item()
        self.grad_norm = torch.norm(all_gradients).item()
        self.grad_mean = torch.mean(all_gradients).item()


    def export_model_graph(self):
        # Dummy input for tracing
        dummy_input = torch.randint(0, self.model_args['vocab_size'], (self.args.batch_size, self.args.block_size)).to(self.device)
        dummy_targets = torch.randint(0, self.model_args['vocab_size'], (self.args.batch_size, self.args.block_size)).to(self.device)  # Dummy targets
        dummy_iter_num = torch.tensor([0], dtype=torch.long).to(self.device) # Dummy iter_num (must be a tensor!)

        # Log the model graph
        if self.args.tensorboard_log and self.args.tensorboard_graph:
            self.writer.add_graph(self.model, (dummy_input, dummy_targets, dummy_iter_num))

        # Export to ONNX and save for Netron
        if self.args.onnx_output:
            onnx_path = os.path.join(self.args.out_dir, "model.onnx")
            torch.onnx.export(self.model,
                              (dummy_input, dummy_targets, dummy_iter_num), # All dummy inputs
                              onnx_path,
                              export_params=True,
                              opset_version=14,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                            'output': {0: 'batch_size', 1: 'sequence_length'}})

    def _loss_bit_statistics(self):
        stats_getter = getattr(self.loss_fn, "bit_statistics", None)
        if not callable(stats_getter):
            return None
        stats = stats_getter()
        if not stats:
            return None
        total_bits = stats.get("total_bits")
        penalty = stats.get("bit_penalty_term")
        if total_bits is None and penalty is None:
            return None
        return stats

    def _log_bit_metrics(self, target_dataset: str, tokens_trained: float) -> None:
        if not self.args.tensorboard_log or self.writer is None:
            return
        stats = self._loss_bit_statistics()
        if not stats:
            return

        total_bits = stats.get("total_bits")
        normalized_bits = stats.get("normalized_bits")
        penalty_term = stats.get("bit_penalty_term")

        if total_bits is not None:
            self.writer.add_scalar(f"{target_dataset}/bit_total_bits", total_bits, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/bit_total_bits_tokens", total_bits, tokens_trained)

        if (
            normalized_bits is not None
            and (total_bits is None or not math.isclose(normalized_bits, total_bits))
        ):
            self.writer.add_scalar(
                f"{target_dataset}/bit_normalized_bits", normalized_bits, self.iter_num
            )
            self.writer.add_scalar(
                f"{target_dataset}/bit_normalized_bits_tokens", normalized_bits, tokens_trained
            )

        if penalty_term is not None:
            self.writer.add_scalar(
                f"{target_dataset}/bit_loss_penalty", penalty_term, self.iter_num
            )
            self.writer.add_scalar(
                f"{target_dataset}/bit_loss_penalty_tokens", penalty_term, tokens_trained
            )

    def log_metrics(self, losses, running_mfu, epoch, tokens_trained, target_dataset, val_better_than_chance):

        if self.iter_num == 0 and self.args.tensorboard_log and self.args.export_model_graph == True  and self.args.compile == False:
            self.export_model_graph()

        if self.args.tensorboard_log:
            # Log metrics for each dataset separately
            self.writer.add_scalars(
                    f"{target_dataset}/loss_iters", {
                        f"val": losses['val'].item(),
                        },
                    self.iter_num
                    )
            self.writer.add_scalars(
                    f"{target_dataset}/loss_tokens", {
                        f"val": losses['val'].item(),
                        },
                    tokens_trained
                    )

            # vocab agnostic, cross tokenizer comparison
            if self.args.log_btc_train:
                self.writer.add_scalars(
                        f"{target_dataset}/chance_tokens",
                        {"val_chance": val_better_than_chance},
                        tokens_trained
                        )
                self.writer.add_scalars(
                        f"{target_dataset}/chance_iters",
                        {"val_chance": val_better_than_chance},
                        self.iter_num
                        )

            # vocab agnostic, cross parameter size comparison
            if self.args.log_btc_per_param:
                self.writer.add_scalars(
                        f"{target_dataset}/btc_per_param_tokens",
                        {"val_chance": val_better_than_chance/self.model.num_param},
                        tokens_trained
                        )
                self.writer.add_scalars(
                        f"{target_dataset}/btc_per_param_iters",
                        {"val_chance": val_better_than_chance/self.model.num_param},
                        self.iter_num
                        )

            self.writer.add_scalar(f"{target_dataset}/epoch", epoch, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/tokens_trained", tokens_trained, self.iter_num)

            self.writer.add_scalar(f"{target_dataset}/vram", self.vram_allocated, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/mfu_pct", running_mfu * 100, self.iter_num)

            self.writer.add_scalar(f"{target_dataset}/loss_vocab", self.model_args['vocab_size'] / torch.exp(losses['val']).item(), self.iter_num)

            self.writer.add_scalar(f"{target_dataset}/lr_iters", self.lr, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/lr_tokens", self.lr, tokens_trained)

            self.writer.add_scalar(f"{target_dataset}/batch_size_iters", self.args.batch_size, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/batch_size_tokens", self.args.batch_size, tokens_trained)

            self.writer.add_scalar(f"{target_dataset}/std_val_iters", losses['val_std'].item(), self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/std_val_tokens", losses['val_std'].item(), tokens_trained)

            if not math.isnan(self.latest_distillation_loss):
                self.writer.add_scalar(
                    f"{target_dataset}/distillation_loss",
                    self.latest_distillation_loss,
                    self.iter_num,
                )

            if 'top1_prob' in losses:
                self.writer.add_scalar(f"{target_dataset}/avg_top1_prob", losses['top1_prob'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/avg_top1_correct", losses['top1_correct'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/avg_target_rank", losses['target_rank'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/avg_target_left_prob", losses['target_left_prob'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/avg_target_prob", losses['target_prob'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/target_rank_95", losses['target_rank_95'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/left_prob_95", losses['left_prob_95'], self.iter_num)
            if 'ln_f_cosine' in losses:
                self.writer.add_scalar(f"{target_dataset}/avg_ln_f_cosine", losses['ln_f_cosine'], self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/ln_f_cosine_95", losses['ln_f_cosine_95'], self.iter_num)

            if self.args.gns_type is not None:
                self.writer.add_scalar(f"{target_dataset}/gns_iters", self.gns, self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/gns_tokens", self.gns, tokens_trained)

            self._log_bit_metrics(target_dataset, tokens_trained)


        if self.args.csv_log:
            # concise training metrics
            self.write_to_csv(target_dataset, losses['train'].item(), losses['val'].item(), prefix=f"{target_dataset}_")

            # bulk metrics
            self.write_to_csv(target_dataset, losses['train'].item(), losses['val'].item(), running_mfu, prefix="bulk_")

    def log_metrics_non_validation(self, loss_training, running_mfu, epoch, tokens_trained, target_dataset, train_better_than_chance):
        if self.args.tensorboard_log:
            self.writer.add_scalars(
                    f"{target_dataset}/loss_iters",
                    {"train": loss_training},
                    self.iter_num
                    )
            self.writer.add_scalars(
                    f"{target_dataset}/loss_tokens",
                    {"train": loss_training},
                    tokens_trained
                    )

            if self.args.log_btc_train:
                self.writer.add_scalars(
                        f"{target_dataset}/chance_tokens",
                        {"train_chance": train_better_than_chance},
                        tokens_trained
                        )
                self.writer.add_scalars(
                        f"{target_dataset}/chance_iters",
                        {"train_chance": train_better_than_chance},
                        self.iter_num
                        )

            if self.args.log_btc_per_param:
                self.writer.add_scalars(
                        f"{target_dataset}/btc_per_param_tokens",
                        {"train_chance": train_better_than_chance/self.model.num_param},
                        tokens_trained
                        )
                self.writer.add_scalars(
                        f"{target_dataset}/btc_per_param_iters",
                        {"train_chance": train_better_than_chance/self.model.num_param},
                        self.iter_num
                        )

            self.writer.add_scalar(f"{target_dataset}/mfu_pct", running_mfu * 100, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/vram", self.vram_allocated, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/param", self.model.num_param, self.iter_num)

            self.writer.add_scalar(f"{target_dataset}/epoch", epoch, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/tokens_trained", tokens_trained, self.iter_num)

            self.writer.add_scalar(f"{target_dataset}/lr_iters", self.lr, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/lr_tokens", self.lr, tokens_trained)

            self.writer.add_scalar(f"{target_dataset}/batch_size_iter", self.args.batch_size, self.iter_num)
            self.writer.add_scalar(f"{target_dataset}/batch_size_tokens", self.args.batch_size, tokens_trained)

            if not math.isnan(self.latest_distillation_loss):
                self.writer.add_scalar(
                    f"{target_dataset}/distillation_loss",
                    self.latest_distillation_loss,
                    self.iter_num,
                )

            if self.args.log_grad_norm:
                self.writer.add_scalar(f"{target_dataset}/grad_norm_iters", self.grad_norm, self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/grad_norm_tokens", self.grad_norm, tokens_trained)

            if self.args.log_grad_std:
                self.writer.add_scalar(f"{target_dataset}/grad_std_iters", self.grad_std, self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/grad_std_tokens", self.grad_std, tokens_trained)

            if self.args.gns_type is not None:
                self.writer.add_scalar(f"{target_dataset}/gns_iters", self.gns, self.iter_num)
                self.writer.add_scalar(f"{target_dataset}/gns_tokens", self.gns, tokens_trained)

            self._log_bit_metrics(target_dataset, tokens_trained)

    def write_to_csv(self, *args, prefix=""):
        args = list(args)
        csv_full_dir = self.args.csv_dir
        if self.args.csv_ckpt_dir:
            csv_full_dir = f"{self.args.csv_dir}/{self.args.csv_ckpt_dir}"
        else:
            if self.args.tensorboard_log:
                sanitized_dataset = self.args.dataset.replace("/", "_")
                csv_full_dir = (
                        f"{self.args.csv_dir}/"
                        f"{sanitized_dataset}_{self.args.tensorboard_run_name}"
                        )
        os.makedirs(csv_full_dir, exist_ok=True)
        # Ensure the filename itself never contains path separators
        safe_csv_name = self.args.csv_name.replace("/", "_")
        csv_path = os.path.join(csv_full_dir, prefix + safe_csv_name + ".csv")

        # `csv_name` is now safe, but make doubly sure every parent dir exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write arguments as a new row in the CSV
            args.insert(0, self.iter_num)
            args.append(self.lr)
            args.append(self.args.batch_size)
            args.append(self.tokens_trained)
            if hasattr(self, "peak_gpu_usage"):
                args.append(self.peak_gpu_usage / (1024 ** 2))
            if self.args.gns_type is not None:
                args.append(self.gns)
            args.append(self.iter_latency_avg)
            writer.writerow(args)


    def log_gamma_beta(self, gamma, beta, layer_num, head_num=None):
        if self.args.tensorboard_log:
            if head_num:
                self.writer.add_scalars(
                        "gammas",
                        {"gamma_L" + str(layer_num) + "_H" + head_num: gamma}, self.iter_num)
                self.writer.add_scalars(
                        "betas",
                        {"beta_L" + str(layer_num) + "_H" + head_num: beta}, self.iter_num)
            else:
                self.writer.add_scalar( "gamma_L" + str(layer_num), gamma, self.iter_num)
                self.writer.add_scalar( "beta_L" + str(layer_num), beta, self.iter_num)

        if self.args.wandb_log and self.master_process:
            import wandb
            wandb.log({
                "iter": self.iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": self.lr,
                "mfu": running_mfu*100,
                })

    def underscore_abbr(self, dataset_name):
        """ Transforms long dataset name to abbreviation
        e.g.
        shakespeare_char -> sc
        commonvoice_ko -> ck
        """
        parts = dataset_name.split('_')
        abbr = ''.join([part[0] for part in parts])
        return abbr

    def save_checkpoint(self, filename):
        if self.args.never_save_checkpoint:
            return
        checkpoint = {
                'model': self.raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'model_args': self.model_args,
                'iter_num': self.iter_num,
                'best_val_loss': self.best_val_loss,
                'best_iter': self.best_iter,
                'best_tokens': self.best_tokens,
                'config': vars(self.args),
                }
        torch.save(checkpoint, os.path.join(self.args.out_dir, filename))

    def train(self):
        if self.args.training_mode == 'multicontext':
            self.X_dict, self.Y_dict, dataset_list = self.get_batch('train')
            current_dataset = dataset_list[0]
            self.mc_btc_train = {}
        else:
            self.X, self.Y, current_dataset = self.get_batch('train')
        self.X, self.Y, current_dataset = self.get_batch('train')
        t_start = time.time()
        t0 = t_start
        local_iter_num = 0
        running_mfu = -1.0
        current_epoch = 0.0
        self.evaluations_remaining = (self.args.max_iters - self.iter_num) // self.args.eval_interval + 1
        self.eta = build_eta_estimator(self.args, t_start, self.evaluations_remaining, self.formatted_completion_eta)
        num_steps_with_worse_loss = 0
        # TODO: Move statistics labels to statistics scripts
        graph_y_labels = []
        for layer in range(self.args.n_layer):
            for head in range(self.args.n_head):
                graph_y_labels.append(f"Layer {layer} Head {head}")

        cli_settings = " ".join(sys.argv)
        cli_text = Text(f"CLI: {cli_settings}", style="chartreuse1")
        self.console = Console()
        # Create progress bar with ETA and remaining time display
        progress = Progress(
                TextColumn("[bold white]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(compact=False),
                TextColumn("-- [bold dark_cyan]BestIter:[/bold dark_cyan]{task.fields[best_iter]} [bold dark_cyan]BestValLoss:[/bold dark_cyan]{task.fields[best_val_loss]} [bold dark_cyan]BestTokens:[/bold dark_cyan]{task.fields[best_tokens]}"),
                TextColumn("-- [bold purple3]ETA:[/bold purple3]{task.fields[eta]}"),
                TextColumn("[bold purple3]Remaining:[/bold purple3]{task.fields[hour]}h{task.fields[min]}m"),
                TextColumn("[bold purple3]total_est:[/bold purple3]{task.fields[total_hour]}h{task.fields[total_min]}m"),
                TextColumn("-- [bold dark_magenta]iter_latency:[/bold dark_magenta]{task.fields[iter_latency]}ms"),
                TextColumn("[bold dark_magenta]peak_gpu_mb:[/bold dark_magenta]{task.fields[peak_gpu_mb]}MB"),
                TextColumn("-- [bold dark_cyan]T1P:[/bold dark_cyan]{task.fields[t1p]}"),
                TextColumn("[bold dark_cyan]T1C:[/bold dark_cyan]{task.fields[t1c]}"),
                TextColumn("-- [bold dark_magenta]TR:[/bold dark_magenta]{task.fields[tr]}"),
                TextColumn("[bold dark_magenta]TP:[/bold dark_magenta]{task.fields[tp]}"),
                TextColumn("[bold dark_magenta]TLP:[/bold dark_magenta]{task.fields[tlp]}"),
                TextColumn("[bold dark_magenta]R95:[/bold dark_magenta]{task.fields[r95]}"),
                TextColumn("[bold dark_magenta]P95:[/bold dark_magenta]{task.fields[p95]}"),
                TextColumn("-- [bold dark_cyan]LnFcos:[/bold dark_cyan]{task.fields[lnf_cos]}"),
                TextColumn("[bold dark_cyan]LnFcos95:[/bold dark_cyan]{task.fields[lnf_cos95]}") ,
                console=self.console
                )

        with Live(Group(progress.get_renderable(), cli_text), console=self.console, refresh_per_second=10) as live:
            task_id = progress.add_task(
                    "[green]Training...",
                    total=((self.args.max_iters - self.iter_num) + self.evaluations_remaining * self.args.eval_iters),
                    eta=self.formatted_completion_eta,
                    total_hour=f"{int(self.total_time_est_ms // 3_600_000)}",
                    total_min=f"{int((self.total_time_est_ms // 60_000) % 60):02d}",
                    hour=f"{int((self.time_remaining_ms // (1000*3600)) % 24):02d}",
                    min=f"{int((self.time_remaining_ms // 60000) % 60):02d}",
                    best_val_loss=f"{self.best_val_loss:.3f}",
                    best_iter=f"{self.best_iter}",
                    best_tokens=f"{self.best_tokens}",
                    iter_latency=f"{self.iter_latency_avg:.1f}",
                    peak_gpu_mb=f"{self.peak_gpu_usage / (1024 ** 2):.1f}",
                    t1p=f"{self.latest_top1_prob:.6f}",
                    t1c=f"{self.latest_top1_correct:.6f}",
                    tr=f"{self.latest_target_rank:.2f}",
                    tp=f"{self.latest_target_prob:.6f}",
                    tlp=f"{self.latest_target_left_prob:.6f}",
                    r95=f"{self.latest_rank_95:.2f}",
                    p95=f"{self.latest_left_prob_95:.6f}",
                    lnf_cos=f"{self.latest_ln_f_cosine:.6f}",
                    lnf_cos95=f"{self.latest_ln_f_cosine_95:.6f}",
                    )

            losses = self.last_eval_losses

            def run_validation_eval():
                nonlocal losses, num_steps_with_worse_loss, running_mfu
                self.update_limit_tracking()
                losses = self.estimate_loss()
                self.last_eval_losses = losses

                self.latest_top1_prob = losses.get('top1_prob', float('nan'))
                self.latest_top1_correct = losses.get('top1_correct', float('nan'))
                self.latest_target_rank = losses.get('target_rank', float('nan'))
                self.latest_target_prob = losses.get('target_prob', float('nan'))
                self.latest_target_left_prob = losses.get('target_left_prob', float('nan'))
                self.latest_rank_95 = losses.get('target_rank_95', float('nan'))
                self.latest_left_prob_95 = losses.get('left_prob_95', float('nan'))
                self.latest_ln_f_cosine = losses.get('ln_f_cosine', float('nan'))
                self.latest_ln_f_cosine_95 = losses.get('ln_f_cosine_95', float('nan'))

                if self.args.gns_type is not None:
                    self.gns = self.gns_ema.get_gns()

                if self.device_type == 'cuda':
                    self.peak_gpu_usage = max(
                            self.peak_gpu_usage,
                            max_memory_allocated(self.device)
                            )

                self.vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                if self.args.dataset_list is not None:
                    # Print loss for each dataset if multiple datasets are used
                    for dataset, dataset_losses in losses['datasets'].items():
                        better_than_chance = self.model_args['vocab_size'] / math.exp(dataset_losses['val'].item())
                        log_message=f"step {self.iter_num}: "
                        log_message+=f"{dataset:<20s}"
                        log_message+=f", {self.model.num_param}"
                        log_message+=f", train loss {dataset_losses['train']:.4f}"
                        log_message+=f", train_stdev {dataset_losses['train_std']:.4f}"
                        log_message+=f", btc_val_set {better_than_chance:.2e}"
                        log_message+=f", btc_val_per_param {(better_than_chance/self.model.num_param):.2e}"
                        log_message+=f", val loss {dataset_losses['val']:.4f}"
                        log_message+=f", val_stdev {dataset_losses['val_std']:.4f}"
                        if self.args.gns_type is not None:
                            log_message+=f", gns {self.gns:.2f}"
                        log_message+=f", lr {self.lr:.4f}"
                        log_message+=f", tokens_trained {self.tokens_trained_dict[dataset]:.2e}"
                        self.console.print(log_message)
                        self.log_metrics(dataset_losses, running_mfu, self.epochs_trained_dict[dataset], self.tokens_trained_dict[dataset], dataset, better_than_chance)
                elif self.args.multicontext_datasets is not None:
                    for dataset, dataset_losses in losses['datasets'].items():
                        log_message=f"step {self.iter_num}: "
                        log_message+=f"{dataset:<20s}"
                        log_message+=f", train loss {dataset_losses['train']:.4f}"
                        log_message+=f", train_stdev {dataset_losses['train_std']:.4f}"
                        log_message+=f", val loss {dataset_losses['val']:.4f}"
                        log_message+=f", val_stdev {dataset_losses['val_std']:.4f}"
                        if self.args.gns_type is not None:
                            log_message+=f", gns {self.gns:.2f}"
                        log_message+=f", lr {self.lr:.4f}"
                        log_message+=f", tokens_trained {self.tokens_trained:.2e}"
                        self.console.print(log_message)
                        better_than_chance = self.vocab_sizes[dataset] / math.exp(dataset_losses['val'].item())
                        self.log_metrics(dataset_losses, running_mfu, current_epoch, self.tokens_trained, dataset, better_than_chance)
                else:
                    # Default behavior for a single dataset
                    better_than_chance = self.model_args['vocab_size'] / math.exp(losses['val'].item())
                    log_message=f"step {self.iter_num}:"
                    log_message+=f", {self.model.num_param}"
                    log_message+=f", train loss {losses['train']:.4f}"
                    log_message+=f", train_stdev {losses['train_std']:.4f}"
                    log_message+=f", btc_val {better_than_chance:.2e}"
                    log_message+=f", btc_val_per_param {(better_than_chance/self.model.num_param):.2e}"
                    log_message+=f", val loss {losses['val']:.4f}"
                    log_message+=f", val_stdev {losses['val_std']:.4f}"
                    if self.args.gns_type is not None:
                        log_message+=f", gns {self.gns:.2f}"
                    log_message+=f", batch_size {self.args.batch_size}"
                    log_message+=f", lr {self.lr:.4f}"
                    self.console.print(log_message)
                    self.log_metrics(losses, running_mfu, current_epoch, self.tokens_trained, current_dataset, better_than_chance)

                if math.isnan(losses["val"]):
                    # If val loss is nan, then exit.
                    with open(self.args.out_dir + "/nan_iter_num.txt", 'w') as file:
                        print("Exiting with nan")
                        file.write(str(self.iter_num))

                if (not self.args.never_save_checkpoint and
                    self.args.save_major_ckpt_interval is not None):
                    if self.iter_num % self.args.save_major_ckpt_interval == 0:
                        major_ckpt_name = str(self.iter_num) +'.pt'
                        # Save major checkpoint
                        self.save_checkpoint(major_ckpt_name)
                        print(f"Saved major checkpoint to {self.args.out_dir}/{major_ckpt_name}")

                if losses['val'] < self.best_val_loss or self.args.always_save_checkpoint:
                    if losses['val'] < self.best_val_loss:
                        self.best_val_loss = losses['val']
                        self.best_iter = self.iter_num
                        self.best_tokens = self.get_limit_tokens_trained()
                        # Save best validation loss
                        peak_mb = self.peak_gpu_usage / (1024 ** 2)
                        with open(os.path.join(self.args.out_dir, 'best_val_loss_and_iter.txt'), "w") as best_loss_file:
                            chance_ratio = self.model_args['vocab_size']/math.exp(self.best_val_loss.item())
                            metrics = [
                                    f"{self.best_val_loss.item()}",
                                    f"{self.iter_num}",
                                    f"{self.best_tokens}",
                                    f"{self.model.num_param}",
                                    f"{chance_ratio:.3e}",
                                    f"{chance_ratio/self.model.num_param:.3e}",
                                    f"{peak_mb:.1f}",
                                    f"{self.iter_latency_avg:.1f}",
                                    f"{losses.get('top1_prob', float('nan')):.6f}",
                                    f"{losses.get('top1_correct', float('nan')):.6f}",
                                    f"{losses.get('target_rank', float('nan')):.2f}",
                                    f"{losses.get('target_left_prob', float('nan')):.6f}",
                                    f"{losses.get('target_prob', float('nan')):.6f}",
                                    f"{losses.get('target_rank_95', float('nan')):.2f}",
                                    f"{losses.get('left_prob_95', float('nan')):.6f}",
                                    f"{losses.get('ln_f_cosine', float('nan')):.6f}",
                                    f"{losses.get('ln_f_cosine_95', float('nan')):.6f}",
                                    f"{self.latest_overall_weight_stats['stdev']:.6f}",
                                    f"{self.latest_overall_weight_stats['kurtosis']:.6f}",
                                    f"{self.latest_overall_weight_stats['max']:.6f}",
                                    f"{self.latest_overall_weight_stats['min']:.6f}",
                                    f"{self.latest_overall_weight_stats['abs_max']:.6f}",
                                    f"{self.latest_overall_activation_stats['stdev']:.6f}",
                                    f"{self.latest_overall_activation_stats['kurtosis']:.6f}",
                                    f"{self.latest_overall_activation_stats['max']:.6f}",
                                    f"{self.latest_overall_activation_stats['min']:.6f}",
                                    f"{self.latest_overall_activation_stats['abs_max']:.6f}",
                            ]
                            best_loss_file.write(", ".join(metrics) + "\n")
                        # Reset early exit counter
                        num_steps_with_worse_loss = 0
                    if self.iter_num > 0 and not self.args.never_save_checkpoint:
                        print(f"saving checkpoint to {self.args.out_dir}")
                        # Save checkpoint
                        self.save_checkpoint('ckpt.pt')

                    # Sample
                    if self.args.max_sample_tokens:
                        live.stop()
                        self.sample_and_print()
                        live.start()
                    # export embedding table to npy file
                    if self.args.export_wte_npy:
                        self.raw_model.export_wte(self.args.export_wte_npy)
                    # export scale matrices to npz file
                    if self.args.export_scale_matrices_npz:
                        self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)
                else:
                    if self.args.sample_each_eval:
                        # Try model inference (e.g. exploring inference from overfitting)
                        if self.args.max_sample_tokens:
                            live.stop()
                            self.sample_and_print()
                            live.start()
                    if self.args.export_wte_each_eval:
                        # export wte table to npy file
                        if self.args.export_wte_npy:
                            self.raw_model.export_wte(self.args.export_wte_npy)
                    if self.args.export_scale_matrices_each_eval:
                        # export scale matrices to npz file
                        if self.args.export_scale_matrices_npz:
                            self.raw_model.export_scale_matrices(self.args.export_scale_matrices_npz)

                if self.args.patience is not None and num_steps_with_worse_loss >= self.args.patience:
                    print(f"Early Stopping: loss has not decreased in {self.args.patience + 1} steps")
                    return True
                if losses['val'] > self.best_val_loss:
                    num_steps_with_worse_loss += 1

                self.tokens_at_last_eval = self.get_limit_tokens_trained()
                self.epochs_at_last_eval = self.limit_epochs_trained
                return False

            while True:
                if self.scheduler is not None:
                    self.lr = self.get_lr(self.iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

                eval_happened = False
                limit_eval_due = self.should_eval_for_limits() if self.master_process else False
                if self.master_process and (
                    (self.iter_num % self.args.eval_interval == 0) or limit_eval_due
                ):
                    should_stop = run_validation_eval()
                    eval_happened = True
                    if should_stop:
                        break

                if self.args.eval_only:
                    break


                for micro_step in range(self.args.gradient_accumulation_steps):
                    if self.ddp:
                        self.model.require_backward_grad_sync = (micro_step == self.args.gradient_accumulation_steps - 1)

                    with self.ctx:
                        if self.args.training_mode == 'multicontext':
                            total_loss = 0
                            logits, training_losses = self.model(
                                None,
                                token_dict=self.X_dict,
                                target_dict=self.Y_dict,
                                iter_num=self.iter_num,
                                loss_fn=self.loss_fn,
                            )

                            # For multicontext training let loss = first dataset loss
                            # loss = training_losses[0]
                            loss = sum(training_losses) / len(training_losses)
                        else:
                            idx_ds = self.args.dataset_list.index(current_dataset) if self.args.dataset_list else None
                            logits, loss = self.model(
                                self.X,
                                targets=self.Y,
                                iter_num=self.iter_num,
                                dataset_idx=idx_ds if self.args.multidataset_wte else None,
                                loss_fn=self.loss_fn,
                            )

                    if hasattr(self.optimizer, "set_entropy") and not isinstance(logits, (list, tuple)):
                        with torch.no_grad():
                            probs = torch.softmax(logits, dim=-1)
                            ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
                            ent = ent / math.log(logits.size(-1))
                        self.optimizer.set_entropy(float(ent))

                    distill_component = None
                    if (
                        self.teacher_model is not None
                        and self.distillation_loss_fn is not None
                        and self.args.training_mode != 'multicontext'
                    ):
                        with torch.no_grad():
                            teacher_logits, _ = self.teacher_model(
                                self.X,
                                targets=self.Y,
                                iter_num=self.iter_num,
                                dataset_idx=idx_ds if self.args.multidataset_wte else None,
                                loss_fn=None,
                            )
                        distill_component = self.distillation_loss_fn(
                            logits,
                            teacher_logits,
                            self.Y,
                            iter_num=self.iter_num,
                        )
                        distill_component = distill_component.to(loss.dtype)
                        loss = loss + self.distillation_weight * distill_component

                    self.latest_distillation_loss = (
                        float(distill_component.detach().float().item())
                        if distill_component is not None
                        else float('nan')
                    )

                    loss = loss / self.args.gradient_accumulation_steps

                    prior_dataset = current_dataset
                    tokens_trained_this_batch = self.args.batch_size * self.args.block_size
                    if self.args.dataset_list:
                        # Update per-dataset count
                        self.tokens_trained_dict[current_dataset] += tokens_trained_this_batch
                        self.tokens_trained = self.tokens_trained_dict[current_dataset]

                        dataset_token_count = self.dataset_size_tokens.get(current_dataset)
                        if dataset_token_count:
                            current_epoch = (
                                self.tokens_trained_dict[current_dataset] / dataset_token_count
                            )
                        else:
                            current_epoch = 0.0
                        self.epochs_trained_dict[current_dataset] = current_epoch
                    else:
                        self.tokens_trained += tokens_trained_this_batch
                        if self.dataset_size_tokens:
                            current_epoch = self.tokens_trained / self.dataset_size_tokens
                        else:
                            current_epoch = 0.0

                    self.update_limit_tracking()

                    self.scaler.scale(loss).backward()

                    # measure grad norms
                    self.get_gradient_stats()

                    if self.args.training_mode == 'multicontext':
                        self.X_dict, self.Y_dict, dataset_list = self.get_batch('train')
                        current_dataset = dataset_list[0]
                    else:
                        self.X, self.Y, current_dataset = self.get_batch('train')

                    if self.args.gns_type is not None:
                        approx_gns_results = gather_hook_results(self.model)
                        self.gns_ema.update(*gns_utils.gnsify(approx_gns_results, self.args.batch_size, ddp=self.ddp))



                if self.args.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                if isinstance(self.optimizer, ActRegularizedAdamW):
                    stat_key = getattr(self.args, "activation_stat", "stdev")
                    self.optimizer.set_activation_stat(
                            self.latest_overall_activation_stats.get(stat_key, 0.0)
                            )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if losses is not None:
                            self.scheduler.step(losses["val"])
                    else:
                        self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                limit_eval_stop = False
                if self.master_process:
                    while self.should_eval_for_limits():
                        should_stop = run_validation_eval()
                        eval_happened = True
                        if should_stop:
                            limit_eval_stop = True
                            break
                if limit_eval_stop:
                    break

                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                self.total_training_time_ms = (t1 - t_start) * 1000.0

                # Estimate ETA
                eta_update: ETAUpdate = self.eta.update(
                        iter_num=self.iter_num,
                        now=t1,
                        dt=dt,
                        is_eval_boundary=eval_happened,
                        )

                progress_advance = eta_update.progress_advance
                self.iter_latency_avg = eta_update.iter_latency_avg
                self.time_remaining_ms = eta_update.time_remaining_ms
                self.formatted_completion_eta = eta_update.formatted_completion_eta



                self.iter_num += 1
                local_iter_num += 1

                if self.iter_num % self.args.log_interval == 0 and self.master_process:
                    lossf = loss.item() * self.args.gradient_accumulation_steps
                    if local_iter_num >= 5:
                        mfu = self.raw_model.estimate_mfu(self.args.batch_size * self.args.gradient_accumulation_steps, dt)
                        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu


                    # training _loss section
                    log_message= f"iter {self.iter_num}"
                    log_message+= f", {dt*1000:.2f} ms"
                    log_message+= f", {self.model.num_param}"
                    if self.args.multicontext_datasets:
                        for i, mc_dataset in enumerate(self.args.multicontext_datasets):
                            self.mc_btc_train[mc_dataset] = self.vocab_sizes[mc_dataset] / math.exp(training_losses[i].item())
                            log_message+= f", {self.underscore_abbr(mc_dataset)}"
                            if self.args.log_btc_train:
                                log_message+= f" btc {self.mc_btc_train[mc_dataset]:.4f}"
                            log_message+= f", {self.underscore_abbr(mc_dataset)}"
                            log_message+= f" loss {training_losses[i].item():.4f}"
                    else:
                        better_than_chance = self.model_args['vocab_size'] / math.exp(lossf)
                        log_message+= f", loss {lossf:.4f}"
                        if self.args.log_btc_train:
                            log_message+=f", btc_train {better_than_chance:.2e}"
                        if self.args.log_btc_per_param:
                            log_message+=f", btc_train_per_param {(better_than_chance/self.model.num_param):.2e}"

                    if self.args.dataset_list:
                        log_message+= f", epoch {self.epochs_trained_dict[prior_dataset]:2.2f}"
                        log_message+= f", tokens_trained {self.tokens_trained_dict[prior_dataset]:.2e}"
                        log_message+= f", dataset: {prior_dataset}"
                    else:
                        log_message+= f", epoch {current_epoch:6.2f}"
                        log_message+= f", tokens_trained {self.tokens_trained:.2e}"
                    log_message+= f", mfu {running_mfu*100:.2f}%"
                    if self.args.gns_type is not None:
                        self.gns = self.gns_ema.get_gns()
                        log_message+= f", gns {self.gns:.2f}"
                    log_message+= f", batch_size {self.args.batch_size}"
                    log_message+= f", lr {self.lr:.4f}"
                    if self.args.log_grad_norm:
                        log_message+= f", grad_norm {self.grad_norm:2f}"
                    if self.args.log_grad_std:
                        log_message+= f", grad_std {self.grad_std:.2f}"

                    self.console.print(log_message)

                    if math.isnan(lossf):
                        # If training loss is nan, then exit.
                        with open(self.args.out_dir + "/nan_iter_num.txt", 'w') as file:
                            file.write(str(self.iter_num))
                            sys.exit("Exiting training loss is NaN")

                    self.vram_allocated = get_gpu_memory_info(info_type='used') if self.args.device != "cpu" else 0
                    if self.args.dataset_list:
                        self.log_metrics_non_validation(lossf, running_mfu, self.epochs_trained_dict[prior_dataset], self.tokens_trained_dict[prior_dataset], prior_dataset, better_than_chance)
                    if self.args.multicontext_datasets:
                        for i, mc_dataset in enumerate(self.args.multicontext_datasets):
                            self.log_metrics_non_validation(training_losses[i].item(), running_mfu, current_epoch, self.tokens_trained, mc_dataset, self.mc_btc_train[mc_dataset])
                    else:
                        self.log_metrics_non_validation(lossf, running_mfu, current_epoch, self.tokens_trained, prior_dataset, better_than_chance)

                if self.args.create_statistics and local_iter_num % self.args.softmax_io_log_interval == 0:
                    create_statistics(self, graph_y_labels)


                # Update progress bar
                self.total_time_est_ms = self.total_training_time_ms + self.time_remaining_ms
                progress.update(
                        task_id,
                        advance=progress_advance,
                        eta=self.formatted_completion_eta,
                        total_hour=f"{int(self.total_time_est_ms // 3_600_000)}",
                        total_min=f"{int((self.total_time_est_ms // 60_000) % 60):02d}",
                        hour=f"{int((self.time_remaining_ms // 3_600_000) % 24):02d}",
                        min=f"{int((self.time_remaining_ms // 60_000) % 60):02d}",
                        best_val_loss=f"{self.best_val_loss:.3f}",
                        best_iter=f"{self.best_iter}",
                        best_tokens=f"{self.best_tokens}",
                        iter_latency=f"{self.iter_latency_avg:.1f}",
                        peak_gpu_mb=f"{self.peak_gpu_usage / (1024 ** 2):.1f}",
                        t1p=f"{self.latest_top1_prob:.6f}",
                        t1c=f"{self.latest_top1_correct:.6f}",
                        tr=f"{self.latest_target_rank:.2f}",
                        tp=f"{self.latest_target_prob:.6f}",
                        tlp=f"{self.latest_target_left_prob:.6f}",
                        r95=f"{self.latest_rank_95:.2f}",
                        p95=f"{self.latest_left_prob_95:.6f}",
                        lnf_cos=f"{self.latest_ln_f_cosine:.6f}",
                        lnf_cos95=f"{self.latest_ln_f_cosine_95:.6f}",
                        )
                live.update(Group(progress.get_renderable(), cli_text))

                # End of training actions
                limit_reached = False
                if self.train_limit_mode in ('epochs', 'tokens'):
                    if self.max_tokens_target is not None:
                        limit_reached = self.limit_tokens_trained >= self.max_tokens_target
                    if not limit_reached and self.max_epochs_target is not None:
                        limit_reached = self.limit_epochs_trained >= self.max_epochs_target
                    if not limit_reached and self.iter_num > self.args.max_iters:
                        limit_reached = True
                else:
                    limit_reached = self.iter_num > self.args.max_iters

                if limit_reached:
                    print(self.best_val_loss, self.best_iter, self.best_tokens)
                    if self.args.only_save_checkpoint_at_end:
                        if not self.args.never_save_checkpoint:
                            self.save_checkpoint('ckpt.pt')
                            print(f"Saved checkpoint to {self.args.out_dir}")

                        # Sample if set
                        if self.args.max_sample_tokens:
                            live.stop()
                            self.sample_and_print()
                            live.start()
                    break

            if self.args.plot_statistics:
                plot_statistics(self.args, self.stats, graph_y_labels)

            if self.args.tensorboard_log:
                self.writer.flush()
                self.writer.close()

            if self.args.wandb_log and self.master_process:
                import wandb
                wandb.log({"finished": True})
                wandb.finish()

def main():
    args, model_group, training_group, logging_group = parse_args()
    trainer = Trainer(args, model_group, training_group, logging_group)

    if not args.sample_only:
        trainer.train()

    if trainer.ddp:
        destroy_process_group()

    if args.tensorboard_log:
        trainer.writer.flush()
        trainer.writer.close()

if __name__ == '__main__':
    main()

