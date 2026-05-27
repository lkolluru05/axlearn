import sys
import subprocess
import re
import os
import json
import threading

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
from tunix import MetricsLoggerOptions
from flax import nnx
from axlearn.common.module import functional as F, install_context_stack

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "gs://axlearn-public/tensorflow_datasets", "Directory containing the TFDS dataset.")
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
    print("lkolluru predicted text: ", text)
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
    for i, completion in enumerate(completions):
        print("lkolluru completion: ", completion)
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
        print("lkolluru expected value: ", expected)
        print("lkolluru predicted value: ", predicted)
        
        
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

def create_tunix_config(mesh, optimizer, ckpt_dir: str):
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=10,
            max_steps=100,
            mini_batch_size=2,
            train_micro_batch_size=2,
            metrics_logging_options=MetricsLoggerOptions(log_dir=ckpt_dir),
            checkpoint_root_directory=ckpt_dir,
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

def create_custom_optimizer(peak_lr: float = 3e-4, max_step: int = 100):
    update_schedule = config_for_function(cosine_with_linear_warmup).set(
        peak_lr=1.0, max_step=max_step, warmup_steps=50, begin_value=0.0, alpha=0.1
    )
    base_optimizer_cfg = config_for_function(chain).set(
        args=[
            config_for_function(clip_by_global_norm).set(max_norm=1.0),
            config_for_function(adamw_decoupled_optimizer).set(
                learning_rate=peak_lr, b1=0.9, b2=0.95, eps=1e-8, update_schedule=update_schedule, weight_decay=0.1
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
        
        input_batch = {"input_ids": input_ids} if not isinstance(input_ids, dict) else input_ids
        state = jax.tree.map(lambda p: getattr(p, "value", p) if isinstance(p, nnx.Variable) else p, self.params)
        
        from axlearn.common.module import _global_context_stack
        if getattr(_global_context_stack, "thread_id", None) != threading.get_ident():
            install_context_stack([])

        (_, aux), _ = F(self.ax_model, is_training=is_training, prng_key=jax.random.PRNGKey(42), state=state, inputs=dict(input_batch=input_batch, return_aux=True))
        outputs = jax.lax.with_sharding_constraint(aux["logits"], jax.sharding.PartitionSpec())
        if len(args) > 2 or "cache" in kwargs or "prefill" in kwargs:
            return outputs, cache
        return outputs

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
        if isinstance(text, bytes): text = text.decode('utf-8')
        
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

    return to_tf_dataset(dataset.skip(1000), True), to_tf_dataset(dataset.take(1000), False)

def evaluate_model(rl_cluster, dataset, rollout_config):
    print("Evaluating model...")
    print("lkolluru tokenizer pad_id:", rl_cluster.tokenizer.pad_id())
    print("lkolluru tokenizer eos_id:", rl_cluster.tokenizer.eos_id())
    print("lkolluru tokenizer bos_id:", rl_cluster.tokenizer.bos_id())
    total_correct, total_samples = 0, 0
    for batch in dataset.take(4):
        prompts = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in batch["prompts"].numpy()]
        print("lkolluru prompts: ",prompts)
        for p in prompts:
            print("lkolluru tokenized prompt (first 50 tokens):", rl_cluster.tokenizer.encode(p)[:50])
        ground_truth = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in batch["ground_truth"].numpy()]
        print("lkolluru ground_truth: ",ground_truth)
        rollout_output = rl_cluster.generate(prompts, rollout_config)
        completions = rollout_output.text
        rewards = math_reward_fn(prompts, completions, ground_truth=ground_truth)
        total_correct += sum(rewards)
        total_samples += len(rewards)
    print(f"Accuracy: {total_correct / max(1, total_samples):.4f}")

def main(_):
    wandb.init()
    trainer_cfg, restored_state, mesh = get_fuji_trainer_and_state(FLAGS.ckpt_dir)
    
    try:
        print("Attempting to load native Llama-3 tokenizer using FujiV3Vocabulary...")
        from axlearn.experiments.text.gpt.vocabulary_fuji_v3 import FujiV3Vocabulary
        raw_tokenizer = FujiV3Vocabulary(filename="Llama-3-tokenizer.json")
        print("Successfully loaded native Llama-3 tokenizer!")
    except Exception as e:
        print(f"Failed to load native Llama-3 tokenizer ({e}). Falling back to HuggingFace...")
        from transformers import AutoTokenizer
        raw_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")

    adapter = LlamaTokenizerAdapter(raw_tokenizer)
    tokenizer = TunixTokenizerWrapper(adapter)
    
    global_batch_size = jax.process_count() * 2
    train_ds, test_ds = prepare_math_dataset(global_batch_size, tokenizer)
    
    optimizer = create_custom_optimizer()
    cluster_config, grpo_config = create_tunix_config(mesh, optimizer, FLAGS.ckpt_dir)
    
    actor_nnx = AXLearnNNXWrapper(trainer_cfg.model, restored_state.model)
    reference_nnx = AXLearnNNXWrapper(trainer_cfg.model, restored_state.model)
    
    rl_cluster = rl_cluster_lib.RLCluster(actor=actor_nnx, reference=reference_nnx, tokenizer=tokenizer, cluster_config=cluster_config)
    
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[math_reward_fn, xml_reward_fn, format_reward_fn],
        algo_config=grpo_config,
    )
    
    evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config)
    grpo_trainer.train(train_ds, test_ds)
    evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config)
    wandb.finish()

if __name__ == "__main__":
    pathwaysutils.initialize()
    app.run(main)