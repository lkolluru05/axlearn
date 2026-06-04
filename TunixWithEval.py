import sys
import subprocess
import re
import os
import json
import threading
from typing import Any, Optional, Tuple

# --- HOT-PATCH FOR FSSPEC GLOB BUG ---
# Forces an upgrade of datasets/fsspec before any other library can cache the broken version.
subprocess.run(
 [sys.executable, "-m", "pip", "install", "-U", "datasets", "huggingface_hub>=0.30.0,<1.0", "fsspec", "kagglehub","--quiet"], 
 check=False
)

# --- HOT-PATCH FOR KAGGLESDK BUG ---
# Fixes ImportError: cannot import name 'get_web_endpoint' from 'kagglesdk.kaggle_env'
try:
    import kagglesdk.kaggle_env
    if not hasattr(kagglesdk.kaggle_env, "get_web_endpoint") and hasattr(kagglesdk.kaggle_env, "get_endpoint"):
        kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint
except ImportError:
    pass

import tensorflow as tf
from absl import app, flags
from axlearn.common import input_tf_data
from axlearn.common.config import config_for_class, config_for_function
import jax
import pathwaysutils
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from axlearn.common.checkpointer import Checkpointer, CheckpointValidationType
from axlearn.common.trainer import TrainerState
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.utils import PartitionSpec, create_device_mesh
from axlearn.experiments.text.gpt.c4_trainer import named_trainer_configs
from axlearn.experiments.text.common import vocab
from axlearn.common.learner import Learner
from axlearn.common.optimizers import adamw_decoupled_optimizer, chain, clip_by_global_norm
from axlearn.common.schedule import cosine_with_linear_warmup
from axlearn.common.optimizer_base import OptParam
from datasets import load_dataset
import optax
import wandb
import numpy as np

# Tunix imports
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.generate import sampler as tunix_sampler
import functools
import operator
from tunix import MetricsLoggerOptions
from tunix.sft.metrics_logger import CluBackend, WandbBackend
from flax import nnx
from axlearn.common.module import functional as F, install_context_stack

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "gs://ericshen-axlearn/tensorflow_datasets", "Directory containing the TFDS dataset.")
flags.DEFINE_string("ckpt_dir", "gs://ericshen-axlearn/checkpoints/llama-3-1-8B-instruct", "Directory containing the checkpoints.")

# XML Tags for Reasoning
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
    wandb.login(key=os.environ["WANDB_API_KEY"])
else:
    print("WANDB_API_KEY not found. Skipping wandb login.")

def extract_math_answer(text):
    """Robustly extract answer from ground truth, handling #### and \boxed{}."""
    if not isinstance(text, str):
        text = str(text)
    if "####" in text:
        text = text.split("####")[-1].strip()
    
    # Extract from \boxed{...} if present
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
    
    return text.strip().replace(",", "")

def extract_predicted_answer(text):
    """Extract answer from model output, prioritizing <answer> tags then \boxed{}."""
    print("lkolluru predicted text: ", repr(text))
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Try <answer> tags (XML format)
    xml_match = re.search(rf'{re.escape(solution_start)}(.*?){re.escape(solution_end)}', text, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip().replace(",", "")
    
    # 2. Try \boxed{...} (LaTeX format)
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
        
    # 3. Fallback to the last number seen
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def math_reward_fn(prompts, completions, **kwargs):
    res = []
    ground_truths = kwargs.get("ground_truth", [])
    
    # Robustly flatten the nested ground truths list/array first
    def flatten_list(item):
        flat = []
        def _recurse(x):
            if hasattr(x, "numpy"): x = x.numpy()
            if hasattr(x, "tolist"):
                try: x = x.tolist()
                except: pass
            if isinstance(x, (list, tuple)) or (hasattr(x, "shape") and len(x.shape) > 0):
                for element in x: _recurse(element)
            else:
                flat.append(x)
        _recurse(item)
        return flat

    ground_truths = flatten_list(ground_truths)

    for i, completion in enumerate(completions):
        if len(ground_truths) > 0:
            gt_idx = i if len(ground_truths) == len(completions) else i // max(1, len(completions) // len(ground_truths))
            gt = ground_truths[gt_idx]
        else:
            gt = ""
        
        if hasattr(gt, "numpy"): gt = gt.numpy()
        if isinstance(gt, bytes): gt = gt.decode('utf-8')
        if hasattr(completion, "numpy"): completion = completion.numpy()
        if isinstance(completion, bytes): completion = completion.decode('utf-8')
        
        expected = extract_math_answer(gt)
        predicted = extract_predicted_answer(completion)
        print(f"lkolluru comparison debug: predicted_repr={repr(predicted)}, expected_repr={repr(expected)}, types=({type(predicted)}, {type(expected)}), match={predicted == expected}", flush=True)
        
        if expected and predicted == expected:
            res.append(1.0)
        else:
            res.append(0.0)
    return res

def xml_reward_fn(prompts, completions, **kwargs):
    """Reward the model for following the <reasoning> and <answer> format."""
    scores = []
    for completion in completions:
        score = 0.0
        # Check for reasoning tags
        if reasoning_start in completion and reasoning_end in completion:
            score += 0.5
        # Check for answer tags
        if solution_start in completion and solution_end in completion:
            score += 0.5
        scores.append(score)
    return scores

def format_reward_fn(prompts, completions, **kwargs):
    """Dense reward for approximate match of the format."""
    scores = []
    for completion in completions:
        score = 0.0
        score += 0.25 if completion.count(reasoning_start) == 1 else -0.1
        score += 0.25 if completion.count(reasoning_end) == 1 else -0.1
        score += 0.25 if completion.count(solution_start) == 1 else -0.1
        score += 0.25 if completion.count(solution_end) == 1 else -0.1
        scores.append(max(0.0, score))
    return scores

def get_fuji_trainer_and_state(ckpt_dir: str):
    ckpt_dir = ckpt_dir.strip()
    trainer_cfg = named_trainer_configs()["fuji-8B-v3-tiktoken"]()
    model_cfg = trainer_cfg.model
    
    checkpointer_cfg = Checkpointer.default_config().set(name="checkpointer", dir=ckpt_dir)
    if hasattr(checkpointer_cfg, "validation_type"):
        checkpointer_cfg.set(validation_type=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE)
    checkpointer = checkpointer_cfg.instantiate(parent=None)
    
    mesh = Mesh(create_device_mesh(mesh_shape=trainer_cfg.mesh_shape), trainer_cfg.mesh_axis_names)
    with mesh:
        model = model_cfg.set(name="model").instantiate(parent=None)
        model_param_specs = model.create_parameter_specs_recursively()
        learner = trainer_cfg.learner.set(name="learner").instantiate(parent=None)
        learner_state_specs = learner.create_state_partition_specs(model_param_specs)
        trainer_state_specs = TrainerState(
            prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
            model=model_param_specs,
            learner=learner_state_specs,
        )

        has_learner = False
        if jax.process_index() == 0:
            try:
                from axlearn.common.checkpointer import read_index_file
                latest_ckpt = Checkpointer.latest_checkpoint_path(ckpt_dir)
                index = read_index_file(latest_ckpt)
                has_learner = any(p.startswith("learner") for p, _ in index)
            except Exception as e:
                print(f"Process 0 failed to read checkpoint index: {e}")
                has_learner = False

        has_learner_arr = jnp.array(1 if has_learner else 0, dtype=jnp.int32)
        has_learner = multihost_utils.broadcast_one_to_all(has_learner_arr).item() == 1

        restore_state_specs = trainer_state_specs._asdict() if has_learner else {"model": model_param_specs}
        step, restored_state_dict = checkpointer.restore(step=None, state=restore_state_specs)
        print("lkolluru checkpoint restored step:", step)

        prng_key = jax.random.PRNGKey(0)
        if step is None:
            restored_state_dict["model"] = model.initialize_parameters_recursively(prng_key=prng_key)

        if "learner" not in restored_state_dict:
            opt_params = jax.tree.map(
                lambda p, spec: OptParam(value=p, factorization_spec=getattr(spec, "factorization", None), weight_decay_scale=getattr(spec, "weight_decay_scale", 1.0)),
                restored_state_dict["model"], model_param_specs
            )
            restored_state_dict["learner"] = learner.init(opt_params)
        
        if "prng_key" not in restored_state_dict:
            restored_state_dict["prng_key"] = prng_key

        restored_state = TrainerState(**{k: v for k, v in restored_state_dict.items() if k in TrainerState._fields})
        return trainer_cfg, restored_state, mesh

def create_tunix_config(mesh, optimizer, model_dir: str):
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine=CustomGreedyRollout,
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=100,
            max_steps=100,
            mini_batch_size=1,
            train_micro_batch_size=1,
            metrics_logging_options=MetricsLoggerOptions(log_dir=model_dir),
            checkpoint_root_directory=model_dir,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=512,
            max_prompt_length=512,
            kv_cache_size=1024 + 256,
            temperature=0.7,
            top_p=0.95,
            top_k=10,
            eos_tokens=[128001],
            return_logprobs=True,
        ),
    )
    grpo_config = GRPOConfig(num_generations=4, num_iterations=1, beta=0.01, epsilon=0.2)
    return cluster_config, grpo_config

from axlearn.common.optimizers import with_partition_fn, opt_param_values
from axlearn.common.optimizer_base import PartitionedGradientTransformation

def adafactor_partitioned(learning_rate: float) -> PartitionedGradientTransformation:
    base = optax.adafactor(learning_rate=learning_rate)
    
    def partition_fn(param_specs):
        dummy_params = jax.tree.map(lambda spec: jnp.zeros(spec.shape, dtype=spec.dtype), param_specs)
        dummy_state = base.init(opt_param_values(dummy_params))
        return jax.tree.map(
            lambda x: OptStateSpec(dtype=x.dtype, shape=x.shape, mesh_axes=PartitionSpec()),
            dummy_state
        )
        
    return with_partition_fn(base, partition_fn)

def create_custom_optimizer(peak_lr: float = 3e-4, max_step: int = 100):
    update_schedule = config_for_function(cosine_with_linear_warmup).set(
        peak_lr=1.0, max_step=max_step, warmup_steps=50, begin_value=0.0, alpha=0.1
    )
    base_optimizer_cfg = config_for_function(chain).set(
        args=[
            config_for_function(clip_by_global_norm).set(max_norm=1.0),
            config_for_function(adafactor_partitioned).set(
                learning_rate=peak_lr
            ),
        ]
    )
    axlearn_opt = base_optimizer_cfg.instantiate()

    def init_fn(params):
        opt_params = jax.tree.map(lambda x: OptParam(value=x, factorization_spec=None, weight_decay_scale=1.0), params)
        return axlearn_opt.init(opt_params)

    def update_fn(updates, state, params=None):
        if params is not None:
            opt_params = jax.tree.map(lambda x: OptParam(value=x, factorization_spec=None, weight_decay_scale=1.0), params)
        else:
            opt_params = None
        return axlearn_opt.update(updates, state, opt_params)

    return optax.GradientTransformation(init_fn, update_fn)

class TunixConfigWrapper:
    def __init__(self, ax_model):
        self.num_layers = 32
        for obj in [ax_model, getattr(ax_model, "decoder", None)]:
            if obj is None: continue
            if hasattr(obj, "config"):
                if hasattr(obj.config, "num_layers"):
                    self.num_layers = obj.config.num_layers
                    break
                if hasattr(obj.config, "layer") and hasattr(obj.config.layer, "num_layers"):
                    self.num_layers = obj.config.layer.num_layers
                    break

        self.num_kv_heads = 8
        for obj in [ax_model, getattr(ax_model, "decoder", None)]:
            if obj is None: continue
            if hasattr(obj, "config"):
                if hasattr(obj.config, "num_kv_heads") and obj.config.num_kv_heads is not None:
                    self.num_kv_heads = obj.config.num_kv_heads
                    break

        self.head_dim = 128
        for obj in [ax_model, getattr(ax_model, "decoder", None)]:
            if obj is None: continue
            if hasattr(obj, "config"):
                if hasattr(obj.config, "head_dim"):
                    self.head_dim = obj.config.head_dim
                    break
                if hasattr(obj.config, "hidden_dim") and hasattr(obj.config, "num_heads"):
                    self.head_dim = obj.config.hidden_dim // obj.config.num_heads
                    break

class AXLearnNNXWrapper(nnx.Module):
    def __init__(self, model_cfg, params):
        self.ax_model = model_cfg.set(name="model").instantiate(parent=None)
        self.params = nnx.data(jax.tree.map(nnx.Param, params))
        self.config = TunixConfigWrapper(self.ax_model)

    def __call__(self, *args, **kwargs):
        input_ids = args[0] if len(args) > 0 else kwargs.get("input_batch", kwargs.get("input_ids"))
        cache = args[2] if len(args) > 2 else kwargs.get("cache")
        is_training = kwargs.get("is_training", False)

        if isinstance(input_ids, dict):
            input_ids = input_ids["input_ids"]

        # Extract the current parameters/state PyTree
        state = jax.tree.map(
            lambda p: (
                getattr(p, "_raw_value", getattr(p, "value", p))
                if isinstance(p, nnx.Variable)
                else p
            ),
            self.params,
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

        # Ensure the functional call context stack is active
        from axlearn.common.module import _global_context_stack
        if getattr(_global_context_stack, "thread_id", None) != threading.get_ident():
            install_context_stack([])

        # Check if we are in prefill (seq_len > 1) or incremental decode (seq_len == 1)
        seq_len = input_ids.shape[-1]

        if seq_len > 1:
            # Prefill Step:
            # 1. Calculate prompt lengths (non-padding tokens)
            # Dynamically detect the pad token ID by checking the repeating token at the start of left-padding
            model_pad_id = self.ax_model.decoder.config.pad_token_id
            is_padded = (input_ids[0, 0] == input_ids[0, 1])
            pad_id = jnp.where(is_padded, input_ids[0, 0], model_pad_id)
            
            # Map/replace all input padding tokens to the model-expected pad token ID so the decoder can mask them
            input_ids = jnp.where(input_ids == pad_id, model_pad_id, input_ids)
                
            non_pad_mask = (input_ids != model_pad_id)
            prompt_lengths = jnp.sum(non_pad_mask, axis=-1, dtype=jnp.int32)
            
            # 2. Convert Left-Padding to Right-Padding (required by AxLearn Decoder)
            # Sort mask so True (non-padding) comes first, then False (padding)
            sort_idx = jnp.argsort(~non_pad_mask, axis=-1, stable=True)
            right_padded_ids = jnp.take_along_axis(input_ids, sort_idx, axis=-1)
            
            # 3. Create input batch for decoder
            input_batch = {
                "input_ids": right_padded_ids,
            }
            
            # Invoke prefill_states functionally on the decoder sub-module
            # Passing prompt_lengths as time_step correctly initializes the KV cache pointers
            (updated_cache, outputs), _ = F(
                module=self.ax_model.decoder,
                inputs=dict(time_step=prompt_lengths, input_batch=input_batch),
                is_training=is_training,
                prng_key=jax.random.PRNGKey(42),
                state=state["decoder"],
                method="prefill_states",
            )

            # Slice the computed logits back to the sampler's static shape (seq_len)
            logits = outputs["logits"][:, :seq_len]
            
            # 4. Un-shift the sliced logits back to the original left-padded layout
            inv_sort_idx = jnp.argsort(sort_idx, axis=-1, stable=True)
            logits = jnp.take_along_axis(logits, jnp.expand_dims(inv_sort_idx, -1), axis=1)
            logits = jax.lax.with_sharding_constraint(logits, jax.sharding.PartitionSpec())
            return logits, updated_cache

        else:
            # Incremental Decode Step:
            # Invoke extend_step functionally on the decoder sub-module
            print("In seq_len == 1 mode")
            input_batch = {
                "input_ids": input_ids,
            }
            (updated_cache, outputs), _ = F(
                module=self.ax_model.decoder,
                inputs=dict(cached_states=cache, input_batch=input_batch),
                is_training=is_training,
                prng_key=jax.random.PRNGKey(42),
                state=state["decoder"],
                method="extend_step",
            )
            
            # Extract logits for the single step
            logits = outputs["logits"]
            logits = jax.lax.with_sharding_constraint(logits, jax.sharding.PartitionSpec())
            return logits, updated_cache

class LlamaTokenizerAdapter:
    def __init__(self, tokenizer_or_vocab):
        self._obj = tokenizer_or_vocab
        self._is_vocab = (
            hasattr(self._obj, "encode")
            and hasattr(self._obj, "decode")
            and not hasattr(self._obj, "convert_tokens_to_ids")
        )
        
        if self._is_vocab:
            self._pad_id = self._obj.pad_id
            self._eos_id = self._obj.eos_id
            self._bos_id = self._obj.bos_id
        else:
            self._bos_id = self._obj.bos_token_id
            if self._bos_id is None or self._bos_id < 0:
                self._bos_id = self._obj.convert_tokens_to_ids("<|begin_of_text|>") or 128000
                
            self._eos_id = self._obj.eos_token_id
            if self._eos_id is None or self._eos_id < 0:
                self._eos_id = self._obj.convert_tokens_to_ids("<|end_of_text|>") or 128001
                
            self._pad_id = self._obj.pad_token_id
            if self._pad_id is None or self._pad_id < 0:
                for candidate in ["<|pad_id|>", "<|finetune_right_pad_id|>", "<|end_of_text|>"]:
                    candidate_id = self._obj.convert_tokens_to_ids(candidate)
                    if candidate_id is not None and candidate_id >= 0:
                        self._pad_id = candidate_id
                        break
                if self._pad_id is None or self._pad_id < 0:
                    self._pad_id = self._eos_id

    def pad_id(self): return self._pad_id
    def eos_id(self): return self._eos_id
    def bos_id(self): return self._bos_id
    def encode(self, text, add_special_tokens=False):
        if self._is_vocab: return self._obj.encode(text)
        return self._obj.encode(text, add_special_tokens=add_special_tokens)
    def decode(self, ids): return self._obj.decode(ids)

class TunixTokenizerWrapper:
    def __init__(self, adapter, max_prompt_length=512):
        self._adapter = adapter
        self.max_prompt_length = max_prompt_length
        
    def pad_id(self): return self._adapter.pad_id()
    def eos_id(self): return self._adapter.eos_id()
    def bos_id(self): return self._adapter.bos_id()
    
    def encode(self, text, *args, **kwargs):
        if hasattr(text, "numpy"): text = text.numpy()
        if hasattr(text, "decode"):
            text = text.decode('utf-8', errors='ignore')
        elif not isinstance(text, str):
            text = str(text)
        
        special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        pattern = "|".join(map(re.escape, special_tokens.keys()))
        
        tokens = []
        last_idx = 0
        for match in re.finditer(pattern, text):
            plain_text = text[last_idx:match.start()]
            if plain_text:
                tokens.extend(self._adapter.encode(plain_text, add_special_tokens=False))
            
            special_tok = match.group(0)
            tokens.append(special_tokens[special_tok])
            last_idx = match.end()
        
        remaining_text = text[last_idx:]
        if remaining_text:
            tokens.extend(self._adapter.encode(remaining_text, add_special_tokens=False))
            
        flat = []
        def _flat(i):
            if isinstance(i, (list, tuple, np.ndarray)): [_flat(x) for x in i]
            else: flat.append(int(i))
        _flat(tokens)
        
        return flat
        
    def decode(self, ids, *args, **kwargs):
        flat = []
        def _flat(i):
            if isinstance(i, (list, tuple, np.ndarray)): [_flat(x) for x in i]
            else: flat.append(int(i))
        _flat(ids)
        return self._adapter.decode(flat)
        
    def apply_chat_template(self, conversation, tokenize=False, **kwargs):
        if isinstance(conversation, str):
            return self.encode(conversation) if tokenize else conversation
            
        if hasattr(self._adapter._obj, "apply_chat_template"):
            try:
                return self._adapter._obj.apply_chat_template(conversation, tokenize=tokenize, **kwargs)
            except Exception as e:
                pass
                
        system_prompt = ""
        user_question = ""
        for msg in conversation:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_question = msg.get("content", "")
                
        text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_question}<|eot_id|>"
        )
        if kwargs.get("add_generation_prompt", True):
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            
        if tokenize:
            return self.encode(text)
        return text
        
    def __getattr__(self, name): return getattr(self._adapter, name)

class CustomRollout(base_rollout.BaseRollout):
    def __init__(
        self,
        *,
        rollout_actor,
        tokenizer,
        mesh,
        rollout_config,
    ):
        kv_cache_size = getattr(rollout_config, "kv_cache_size", 2048)
        self._model = rollout_actor
        self._tokenizer = tokenizer
        self._pad_id = tokenizer.pad_id()
        self._eos_id = tokenizer.eos_id()
        
        cache_config = tunix_sampler.CacheConfig(
            cache_size=kv_cache_size,
            num_layers=rollout_actor.config.num_layers,
            num_kv_heads=rollout_actor.config.num_kv_heads,
            head_dim=rollout_actor.config.head_dim,
        )
        
        self._sampler = tunix_sampler.Sampler(
            rollout_actor,
            tokenizer,
            cache_config,
        )
        self._mesh = mesh
        self._rollout_config = rollout_config

    def generate(
        self,
        prompts: list[str],
        rollout_config: base_rollout.RolloutConfig,
        **kwargs,
    ) -> base_rollout.RolloutOutput:
        import inspect
        sig = inspect.signature(self._sampler.__call__)
        sampler_kwargs = {
            "input_strings": prompts,
            "max_generation_steps": rollout_config.max_tokens_to_generate,
            "max_prompt_length": rollout_config.max_prompt_length,
            "echo": False,
            "temperature": rollout_config.temperature,
            "top_p": rollout_config.top_p,
            "top_k": rollout_config.top_k,
            "seed": rollout_config.seed,
            "pad_output": False,
            "eos_tokens": rollout_config.eos_tokens,
        }
        if "return_logprobs" in sig.parameters:
            sampler_kwargs["return_logprobs"] = rollout_config.return_logprobs
            
        with self._mesh:
            output = self._sampler(**sampler_kwargs)
        logprobs = getattr(output, "logprobs", None)
        
        return base_rollout.RolloutOutput(
            text=output.text,
            logits=output.logits,
            tokens=output.tokens,
            left_padded_prompt_tokens=output.padded_prompt_tokens,
            logprobs=logprobs,
        )

    def get_per_token_logps(
        self,
        prompt_tokens: jax.Array,
        completion_tokens: jax.Array,
        completion_mask: jax.Array | None = None,
    ) -> jax.Array:
        from tunix.rl import common as rl_common
        graphdef, state = self._sampler.model_def_and_state()
        return rl_common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pad_id=self.pad_id(),
            eos_id=self.eos_id(),
            completion_mask=completion_mask,
            stop_gradient=True,
            return_logits=False,
        )

    def update_params(
        self,
        params: Any,
        filter_types: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        from tunix.rl import reshard as rl_reshard
        from tunix.rl import utils as rl_utils
        
        if filter_types is not None:
            dst_params = nnx.state(self.model(), filter_types)
            resharded_params = rl_reshard.reshard_pytree(params, dst_params)
        else:
            resharded_params = params
            
        flat_new_params, _ = rl_utils.to_flat_dict(resharded_params)
        new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
        rollout_precision = jax.tree.leaves(self._sampler.transformer_state)[0].dtype
        
        if new_params_precision != rollout_precision:
            flat_new_params = jax.tree.map(
                lambda x: x.astype(rollout_precision), flat_new_params
            )
            
        flat_old_params, tree_def = rl_utils.to_flat_dict(self._sampler.transformer_state)
        merged_params = functools.reduce(
            operator.ior, [flat_old_params, flat_new_params], {}
        )
        merged_params = jax.tree.unflatten(tree_def, merged_params.values())
        new_model = nnx.merge(self._sampler._transformer_graphdef, merged_params)
        self._sampler.transformer_state = nnx.variables(new_model, nnx.Param)

    def pad_id(self) -> int: return self._pad_id
    def eos_id(self) -> int: return self._eos_id
    def model(self) -> Any: return self._model
    
    @property
    def mesh(self): return self._mesh

class CustomVllmRollout(base_rollout.BaseRollout):
    def __init__(
        self,
        *,
        rollout_actor,
        tokenizer,
        mesh,
        rollout_config,
    ):
        from tunix.generate import vllm_sampler
        from tunix.generate import mappings
        
        mapping_config = mappings.MappingConfig.build(
            mapping_obj=rollout_config.rollout_mapping_config,
            model=rollout_actor,
            backend="vllm_jax",
        )
        
        self._model = rollout_actor
        self._tokenizer = tokenizer
        self._pad_id = tokenizer.pad_id()
        self._eos_id = tokenizer.eos_id()
        
        max_model_len = getattr(rollout_config, "kv_cache_size", 2048)
        
        self._sampler = vllm_sampler.VllmSampler(
            tokenizer=tokenizer,
            config=vllm_sampler.VllmConfig(
                server_mode=rollout_config.rollout_vllm_server_mode,
                mapping_config=mapping_config,
                return_logprobs=rollout_config.return_logprobs,
                init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
                tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
                additional_config=rollout_config.rollout_vllm_additional_config,
                enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
                hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
                lora_config=rollout_config.rollout_vllm_lora_config,
                mesh=mesh,
                tensor_parallel_size=rollout_config.tensor_parallel_size,
                data_parallel_size=rollout_config.data_parallel_size,
                expert_parallel_size=rollout_config.expert_parallel_size,
                delete_dst_buffers=rollout_config.rollout_vllm_delete_dst_buffers,
                reshard_chunk_size=rollout_config.rollout_vllm_reshard_chunk_size,
                engine_kwargs={
                    "model": rollout_config.rollout_vllm_model_version,
                    "max_model_len": max_model_len,
                    "async_scheduling": rollout_config.rollout_vllm_async_scheduling,
                    "max_num_batched_tokens": rollout_config.rollout_vllm_max_num_batched_tokens,
                    "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                    "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                    "max_logprobs": 1,
                    "logprobs_mode": rollout_config.rollout_vllm_logprobs_mode,
                    **rollout_config.rollout_vllm_kwargs,
                },
                sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
            ),
        )
        state = nnx.state(rollout_actor)
        self._sampler.load_checkpoint(state)
        self._mesh = mesh

    @property
    def mesh(self) -> jax.sharding.Mesh:
        return self._sampler.mesh

    def generate(
        self,
        prompts: list[str],
        rollout_config: base_rollout.RolloutConfig,
        **kwargs,
    ) -> base_rollout.RolloutOutput:
        self.output = self._sampler(
            input_strings=prompts,
            max_generation_steps=rollout_config.max_tokens_to_generate,
            max_prompt_length=rollout_config.max_prompt_length,
            temperature=rollout_config.temperature,
            top_p=rollout_config.top_p,
            top_k=rollout_config.top_k,
            seed=rollout_config.seed,
            echo=False,
            pad_output=True,
            **kwargs,
        )

        return base_rollout.RolloutOutput(
            text=self.output.text,
            logits=None,
            tokens=self.output.tokens,
            left_padded_prompt_tokens=self.output.padded_prompt_tokens,
            logprobs=self.output.logprobs,
        )

    def get_per_token_logps(
        self,
        prompt_tokens: jax.Array,
        completion_tokens: jax.Array,
        completion_mask: jax.Array | None = None,
    ) -> jax.Array:
        return self.output.logprobs

    def update_params(
        self,
        params: Any,
        filter_types: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        self._sampler.update_params(params, filter_types)

    def pad_id(self) -> int: return self._pad_id
    def eos_id(self) -> int: return self._eos_id
    def model(self) -> Any: return self._model

class CustomGreedyRollout(base_rollout.BaseRollout):
    def __init__(
        self,
        *,
        rollout_actor,
        tokenizer,
        mesh,
        rollout_config,
    ):
        self._model = rollout_actor
        self._tokenizer = tokenizer
        self._pad_id = tokenizer.pad_id()
        self._eos_id = tokenizer.eos_id()
        self._mesh = mesh
        self._rollout_config = rollout_config
        self._model_def, _ = nnx.split(rollout_actor)

        # JIT compile functionally once to prevent recompilation on every call.
        @jax.jit
        def forward_step(state, cache, ids):
            merged_model = nnx.merge(self._model_def, state)
            return merged_model(ids, cache=cache)
        self._forward_step = forward_step

    def generate(
        self,
        prompts: list[str],
        rollout_config: base_rollout.RolloutConfig,
        **kwargs,
    ) -> base_rollout.RolloutOutput:
        max_new_tokens = rollout_config.max_tokens_to_generate or 128
        # 1. Determine the original shape of prompts (e.g. [B, group_size] or [B])
        original_B = len(prompts)
        if isinstance(prompts[0], (list, tuple)) or hasattr(prompts[0], "shape"):
            group_size = len(prompts[0])
        else:
            group_size = 1
            
        # 2. Recursively flatten any nested lists/arrays of strings into a flat 1D list of strings
        def flatten_strings(item):
            flat = []
            def _recurse(x):
                if hasattr(x, "numpy"): x = x.numpy()
                if isinstance(x, bytes): x = x.decode('utf-8')
                if isinstance(x, (list, tuple)) or (hasattr(x, "shape") and len(x.shape) > 0):
                    for element in x: _recurse(element)
                else:
                    flat.append(str(x))
            _recurse(item)
            return flat
            
        flat_prompts = flatten_strings(prompts)
        print(f"Flattened prompts count: {len(flat_prompts)}")
        print(f"First flattened prompt: {repr(flat_prompts[0])}")
        max_prompt_limit = getattr(self._rollout_config, "max_prompt_length", 512) or 512
        tokens_list = [self._tokenizer.encode(p)[:max_prompt_limit] for p in flat_prompts]
        B = len(flat_prompts)  # Total individual prompts count (B * group_size)
        print(f"Tokenized {B} prompts. First token list length: {len(tokens_list[0])}")
        
        # Calculate padded batch size to match TPU mesh multiple
        multiple = self._mesh.devices.size
        import math
        padded_B = int(math.ceil(B / multiple) * multiple)
        print(f"Batch size: {B}, Padded batch size: {padded_B} (mesh multiple: {multiple})")
        
        max_prompt_len = max_prompt_limit
        max_len = max_prompt_len + max_new_tokens
        input_ids_np = np.full((padded_B, max_len), self.pad_id(), dtype=np.int32)
        print(f"Created input_ids_np with shape: {input_ids_np.shape}")
        
        def flatten_list(item):
            flat = []
            def _recurse(x):
                if hasattr(x, "numpy"): x = x.numpy()
                if hasattr(x, "tolist"):
                    try: x = x.tolist()
                    except: pass
                if isinstance(x, (list, tuple)) or (hasattr(x, "shape") and len(x.shape) > 0):
                    for element in x: _recurse(element)
                else:
                    try: flat.append(int(x))
                    except: pass
            _recurse(item)
            return flat

        # 5. Build Prefill input_ids and paddings (0 = valid, 1 = padded)
        prefill_ids_np = np.full((padded_B, max_prompt_len), self.pad_id(), dtype=np.int32)
        prefill_paddings_np = np.ones((padded_B, max_prompt_len), dtype=np.int32)
        
        for b, tokens in enumerate(tokens_list):
            tokens = flatten_list(tokens)
            length = len(tokens)
            start_idx = max_prompt_len - length
            prefill_ids_np[b, start_idx:max_prompt_len] = tokens
            prefill_paddings_np[b, start_idx:max_prompt_len] = 0
            
        # Replicate first prompt to pad the batch from B to padded_B
        for b in range(B, padded_B):
            tokens = flatten_list(tokens_list[0])
            length = len(tokens)
            start_idx = max_prompt_len - length
            prefill_ids_np[b, start_idx:max_prompt_len] = tokens
            prefill_paddings_np[b, start_idx:max_prompt_len] = 0
            
        prefill_ids = jnp.array(prefill_ids_np)
        prefill_paddings = jnp.array(prefill_paddings_np)
        
        generated_tokens_batch = [[] for _ in range(padded_B)]
        state = nnx.state(self._model)
        cache = None
        
        # 6. Prefill Phase: processes the full prompt length at once to compute initial cache
        prefill_batch = {
            "input_ids": prefill_ids,
            "paddings": prefill_paddings,
        }
        with self._mesh:
            logits, cache = self._forward_step(state, cache, prefill_batch)
            
        # Extract the first generated tokens from the last logit of the prefill phase
        next_token_logits = jax.block_until_ready(logits[:, -1, :])
        next_tokens = jax.block_until_ready(jnp.argmax(next_token_logits, axis=-1))
        next_tokens_np = np.array(next_tokens)
        
        for b in range(padded_B):
            generated_tokens_batch[b].append(int(next_tokens_np[b]))
            
        # 7. Decode Phase: step-by-step decoding with single token inputs
        curr_tokens = jnp.expand_dims(next_tokens, axis=-1)  # Shape: (padded_B, 1)
        decode_paddings = jnp.zeros((padded_B, 1), dtype=np.int32)
        
        for step in range(1, max_new_tokens):
            decode_batch = {
                "input_ids": curr_tokens,
                "paddings": decode_paddings,
            }
            with self._mesh:
                logits, cache = self._forward_step(state, cache, decode_batch)
                
            next_token_logits = logits[:, 0, :]
            next_tokens = jnp.argmax(next_token_logits, axis=-1)
            next_tokens_np = np.array(next_tokens)
            
            for b in range(padded_B):
                generated_tokens_batch[b].append(int(next_tokens_np[b]))
                
            curr_tokens = jnp.expand_dims(next_tokens, axis=-1)
            
        tokens_np = np.full((B, max_new_tokens), self.pad_id(), dtype=np.int32)
        generated_texts = []
        
        for b in range(B):
            truncated_tokens = []
            for idx, t in enumerate(generated_tokens_batch[b]):
                if t == self.eos_id() or t == 128009:
                    break
                tokens_np[b, idx] = t
                truncated_tokens.append(t)
                
            decoded_text = self._tokenizer.decode(truncated_tokens)
            generated_texts.append(decoded_text)
            
        padded_prompt_tokens_list = [prefill_ids_np[b, :max_prompt_len] for b in range(B)]
        
        return base_rollout.RolloutOutput(
            text=generated_texts,
            logits=None,
            tokens=tokens_np,
            left_padded_prompt_tokens=np.array(padded_prompt_tokens_list),
            logprobs=None,
        )

    def get_per_token_logps(
        self,
        prompt_tokens: jax.Array,
        completion_tokens: jax.Array,
        completion_mask: jax.Array | None = None,
    ) -> jax.Array:
        from tunix.rl import common as rl_common
        graphdef, state = nnx.split(self._model)
        return rl_common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            pad_id=self.pad_id(),
            eos_id=self.eos_id(),
            completion_mask=completion_mask,
            stop_gradient=True,
            return_logits=False,
        )

    def update_params(
        self,
        params: Any,
        filter_types: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        if filter_types is not None:
            from tunix.rl import reshard as rl_reshard
            dst_params = nnx.state(self._model, filter_types)
            resharded_params = rl_reshard.reshard_pytree(params, dst_params)
        else:
            resharded_params = params
        nnx.update(self._model, resharded_params)

    def pad_id(self) -> int: return self._pad_id
    def eos_id(self) -> int: return self._eos_id
    def model(self) -> Any: return self._model
    @property
    def mesh(self): return self._mesh

def prepare_math_dataset(batch_size: int, tokenizer, dataset_name="nvidia/OpenMathInstruct-1"):
    print(f"Loading {dataset_name} dataset...")
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True, streaming=True)
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    
    def format_example(example):
        full_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example.get('question', '')},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "prompts": full_prompt,
            "ground_truth": example.get('expected_answer', '')
        }
    
    dataset = dataset.map(format_example)
    
    def to_tf_dataset(hf_ds, is_training):
        def generator():
            for row in hf_ds:
                yield {"prompts": row["prompts"], "ground_truth": row["ground_truth"]}
        output_signature = {"prompts": tf.TensorSpec((), tf.string), "ground_truth": tf.TensorSpec((), tf.string)}
        tf_ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if is_training: tf_ds = tf_ds.shuffle(1000)
        return tf_ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    # Use a very small number of examples for quick training and evaluation
    eval_dataset = dataset.take(16)
    train_dataset = dataset.skip(16).take(32)
    
    return to_tf_dataset(train_dataset, True), to_tf_dataset(eval_dataset, False)

def evaluate_model(rl_cluster, dataset, rollout_config):
    print("Evaluating model...")
    print("lkolluru tokenizer pad_id:", rl_cluster.tokenizer.pad_id())
    print("lkolluru tokenizer eos_id:", rl_cluster.tokenizer.eos_id())
    print("lkolluru tokenizer bos_id:", rl_cluster.tokenizer.bos_id())
    total_correct, total_samples = 0, 0
    for batch in dataset.take(4):
        prompts = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in batch["prompts"].numpy()]
        print("lkolluru prompts from eval: ", prompts[:1]) # Only print first to avoid log clutter
        ground_truth = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in batch["ground_truth"].numpy()]
        print("lkolluru ground_truth from eval: ", ground_truth[:1])
        
        rollout_output = rl_cluster.rollout.generate(prompts, rollout_config)
        completions = rollout_output.text
        print("lkolluru completion from eval (first of batch): ", repr(completions[0]))
        
        rewards = math_reward_fn(prompts, completions, ground_truth=ground_truth)
        total_correct += sum(rewards)
        total_samples += len(rewards)
    print(f"Accuracy: {total_correct / max(1, total_samples):.4f}")

def custom_wandb_and_console_logger(metrics_buffer):
    log_dict = {}
    for metric_name, (values, op) in metrics_buffer.metrics.items():
        agg_value = np.array(values)
        if agg_value.size > 0:
            if op is not None:
                agg_value = op(agg_value)
            else:
                agg_value = np.mean(agg_value)
            
            log_dict[f"{metrics_buffer.mode}/{metric_name}"] = float(agg_value)
            
    if len(log_dict) > 0:
        wandb.log(log_dict, step=metrics_buffer.global_steps)
        if jax.process_index() == 0:
            pct = (metrics_buffer.global_steps / max(1, metrics_buffer.max_global_steps if hasattr(metrics_buffer, "max_global_steps") else 100)) * 100
            print(f"Actor Training: {pct:.1f}% | Step {metrics_buffer.global_steps} ({metrics_buffer.mode})")
            for k, v in log_dict.items():
                print(f"  {k}: {v:.6f}")

def main(_):
    wandb.init()
    trainer_cfg, restored_state, mesh = get_fuji_trainer_and_state(FLAGS.ckpt_dir)
    
    print("Forcing TikToken vocabulary alignment using HuggingFace AutoTokenizer...")
    from transformers import AutoTokenizer
    raw_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")

    adapter = LlamaTokenizerAdapter(raw_tokenizer)
    tokenizer = TunixTokenizerWrapper(adapter)
    
    global_batch_size = max(16, jax.process_count() * 2)
    train_ds, test_ds = prepare_math_dataset(global_batch_size, tokenizer)
    
    optimizer = create_custom_optimizer()
    model_dir = os.path.join(FLAGS.ckpt_dir, "rloutputs")
    cluster_config, grpo_config = create_tunix_config(mesh, optimizer, model_dir)
    
    with mesh:
        actor_nnx = AXLearnNNXWrapper(trainer_cfg.model, restored_state.model)
        reference_nnx = AXLearnNNXWrapper(trainer_cfg.model, restored_state.model)
        
        rl_cluster = rl_cluster_lib.RLCluster(actor=actor_nnx, reference=reference_nnx, tokenizer=tokenizer, cluster_config=cluster_config)
        rl_cluster.with_external_metrics_logger(custom_wandb_and_console_logger)
    
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[math_reward_fn, xml_reward_fn, format_reward_fn],
        algo_config=grpo_config,
    )
    
    # Monkey-patch the reward manager's prompts list to duplicate them G times automatically
    # to bypass the framework length validation check cleanly!
    original_compute_rewards = grpo_trainer.reward_manager._compute_rewards
    
    def patched_compute_rewards(prompts, completions, **kwargs):
        if len(prompts) != len(completions) and len(completions) % len(prompts) == 0:
            group_size = len(completions) // len(prompts)
            duplicated_prompts = []
            for p in prompts:
                duplicated_prompts.extend([p] * group_size)
            prompts = duplicated_prompts
            
        m_rewards = math_reward_fn(prompts, completions, **kwargs)
        x_rewards = xml_reward_fn(prompts, completions, **kwargs)
        f_rewards = format_reward_fn(prompts, completions, **kwargs)

        m_arr = np.array(m_rewards, dtype=np.float32)
        x_arr = np.array(x_rewards, dtype=np.float32)
        f_arr = np.array(f_rewards, dtype=np.float32)
        total_arr = m_arr + x_arr + f_arr

        print(f"Explicit Math Reward Mean: {np.mean(m_arr):.4f} (Sum: {np.sum(m_arr)})", flush=True)
        print(f"Explicit XML Reward Mean: {np.mean(x_arr):.4f}, Format: {np.mean(f_arr):.4f}", flush=True)

        log_metrics = {
            "rewards/math_reward_fn": (float(np.mean(m_arr)), None),
            "rewards/xml_reward_fn": (float(np.mean(x_arr)), None),
            "rewards/format_reward_fn": (float(np.mean(f_arr)), None),
            "rewards/sum": (float(np.mean(total_arr)), None),
        }

        if hasattr(grpo_trainer.reward_manager, "metrics"):
            grpo_trainer.reward_manager.metrics.update(log_metrics)

        return {
            "rewards": jnp.array(total_arr, dtype=jnp.float32),
            "log_metrics": log_metrics,
        }
        
    grpo_trainer.reward_manager._compute_rewards = patched_compute_rewards
    
    #evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config)
    grpo_trainer.train(train_ds, test_ds)
    evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config)
    wandb.finish()

if __name__ == "__main__":
    pathwaysutils.initialize()
    app.run(main)