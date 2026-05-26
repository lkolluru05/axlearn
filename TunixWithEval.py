import sys
import subprocess

# --- HOT-PATCH FOR FSSPEC GLOB BUG ---
# Forces an upgrade of datasets/fsspec before any other library can cache the broken version.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-U", "datasets", "huggingface_hub", "fsspec", "--quiet"], 
    check=False
)

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
from axlearn.common.update_transformation import ConditionalUpdateTransformation
import threading
import optax
from axlearn.common.optimizer_base import OptParam
import re
import os
import json
from datasets import load_dataset

# Tunix imports
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix import MetricsLoggerOptions
from flax import nnx
from axlearn.common.module import functional as F, install_context_stack, ContextStack
from axlearn.common.module import InvocationContext, new_output_collection, set_current_context

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "gs://axlearn-public/tensorflow_datasets", "Directory containing the TFDS dataset.")
flags.DEFINE_string("ckpt_dir", "gs://ericshen-axlearn/checkpoints/llama-3-1-8B-instruct", "Directory containing the checkpoints.")

import wandb
import os
  # Check if WANDB_API_KEY is set before logging inf
if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
      wandb.login(key=os.environ["WANDB_API_KEY"])
else:
      print("WANDB_API_KEY not found. Skipping wandb login.")

def extract_prompts() -> input_tf_data.DatasetToDatasetFn:
    def process_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
        def map_fn(features):
            features = dict(features)
            features["prompts"] = features["text"]
            return features
        return ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return process_fn

def get_dataset_config(is_training: bool, dataset_name: str, split: str, batch_size: int = 32, data_dir: str = None):
    read_config = config_for_function(input_tf_data.tfds_read_config)
    source_cfg = config_for_function(input_tf_data.tfds_dataset).set(
        dataset_name=dataset_name,
        split=split,
        is_training=is_training,
        read_config=read_config,
        data_dir=data_dir,
        train_shuffle_buffer_size=1024 * 16,
    )
    processor_cfg = config_for_function(extract_prompts)
    batcher_cfg = config_for_function(input_tf_data.batch).set(
        global_batch_size=batch_size,
        pad_example_fn=input_tf_data.default_pad_example_fn,
    )
    return input_tf_data.Input.default_config().set(
        name="input",
        is_training=is_training,
        source=source_cfg,
        processor=processor_cfg,
        batcher=batcher_cfg,
    )

def get_dataset(is_training: bool, dataset_name: str, split: str, batch_size: int = 32, data_dir: str = None):
    input_cfg = get_dataset_config(is_training, dataset_name, split, batch_size, data_dir)
    input_module = input_cfg.instantiate(parent=None)
    return input_module.dataset()

def get_fuji_trainer_and_state(ckpt_dir: str):
    ckpt_dir = ckpt_dir.strip()
    trainer_cfg = named_trainer_configs()["fuji-8B-v3-tiktoken"]()
    model_cfg = trainer_cfg.model
    print("Successfully loaded Fuji trainer config.")

    checkpointer_cfg = Checkpointer.default_config().set(name="checkpointer", dir=ckpt_dir)
    if hasattr(checkpointer_cfg, "validation_type"):
        checkpointer_cfg.set(validation_type=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE)
    checkpointer = checkpointer_cfg.instantiate(parent=None)
    mesh = Mesh(create_device_mesh(mesh_shape=trainer_cfg.mesh_shape), trainer_cfg.mesh_axis_names)
    print("lkolluru mesh: ", mesh)
    with mesh:
        model = model_cfg.set(name="model").instantiate(parent=None)
        print("lkolluru model: ", model)
        model_param_specs = model.create_parameter_specs_recursively()
        print("lkolluru model_param_specs: ", model_param_specs)
        learner = trainer_cfg.learner.set(name="learner").instantiate(parent=None)
        learner_state_specs = learner.create_state_partition_specs(model_param_specs)
        print("lkolluru learner_state_specs: ", learner_state_specs)
        trainer_state_specs = TrainerState(
            prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
            model=model_param_specs,
            learner=learner_state_specs,
        )
        print("lkolluru trainer_state_specs: ", trainer_state_specs)
        
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
                
        # Broadcast `has_learner` to all workers to prevent distributed deadlocks
        has_learner_arr = jnp.array(1 if has_learner else 0, dtype=jnp.int32)
        has_learner = multihost_utils.broadcast_one_to_all(has_learner_arr).item() == 1

        restore_state_specs = trainer_state_specs._asdict() if has_learner else {"model": model_param_specs}
        step, restored_state_dict = checkpointer.restore(step=None, state=restore_state_specs)
        print("lkolluru step: ", step)
        
        prng_key = jax.random.PRNGKey(0)
        if step is None:
            print("Checkpoint not found, initializing model parameters from scratch.")
            restored_state_dict["model"] = model.initialize_parameters_recursively(prng_key=prng_key)
            
        if "learner" not in restored_state_dict:
            opt_params = jax.tree.map(
                lambda p, spec: OptParam(
                    value=p,
                    factorization_spec=getattr(spec, "factorization", None),
                    weight_decay_scale=getattr(spec, "weight_decay_scale", 1.0)
                ),
                restored_state_dict["model"],
                model_param_specs
            )
            if hasattr(learner, "init_states"):
                restored_state_dict["learner"] = learner.init_states(opt_params)
            elif hasattr(learner, "initialize_state"):
                restored_state_dict["learner"] = learner.initialize_state(opt_params)
            elif hasattr(learner, "init_state"):
                restored_state_dict["learner"] = learner.init_state(opt_params)
            else:
                restored_state_dict["learner"] = learner.init(opt_params)
        if "prng_key" not in restored_state_dict:
            restored_state_dict["prng_key"] = prng_key

        restored_state = TrainerState(**{k: v for k, v in restored_state_dict.items() if k in TrainerState._fields})
        
    print(f"Restored state from step: {step}")
    return trainer_cfg, restored_state, mesh

def dummy_reward_fn(prompts, completions, **kwargs):
    res=[]
    for completion in completions:
        if ",,,," not in completion:
            res.append(1.0)
        else:
            res.append(0.5)
    return res

def extract_math_answer(text):
    if not isinstance(text, str):
        text = str(text)
    if "####" in text:
        return text.split("####")[-1].strip().replace(",", "")
    return ""

def extract_predicted_answer(text):
    if not isinstance(text, str):
        text = str(text)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def math_reward_fn(prompts, completions, **kwargs):
    res = []
    ground_truths = kwargs.get("ground_truth", [])
    for i, completion in enumerate(completions):
        if len(ground_truths) > 0:
            gt_idx = i if len(ground_truths) == len(completions) else i // max(1, len(completions) // len(ground_truths))
            gt = ground_truths[gt_idx]
        else:
            gt = ""
        if hasattr(gt, "numpy"):
            gt = gt.numpy()
        if isinstance(gt, bytes):
            gt = gt.decode('utf-8')
        if hasattr(completion, "numpy"):
            completion = completion.numpy()
        if isinstance(completion, bytes):
            completion = completion.decode('utf-8')
        expected = extract_math_answer(gt)
        predicted = extract_predicted_answer(completion)
        if expected and predicted == expected:
            res.append(1.0)
        else:
            res.append(0.0)
    return res

def create_tunix_config(mesh, optimizer, ckpt_dir: str):
    EVAL_EVERY_N_STEPS = 10
    MAX_STEPS = 100
    TRAIN_MICRO_BATCH_SIZE = 2
    TOTAL_GENERATION_STEPS = 256
    MAX_PROMPT_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TOP_K = 10
    EOS_TOKENS = [128001]
    NUM_GENERATIONS = 4
    NUM_ITERATIONS = 1
    BETA = 0.01
    EPSILON = 0.2

    metrics_logging_options = MetricsLoggerOptions(
        log_dir=ckpt_dir,
    )

    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=MAX_STEPS,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=ckpt_dir,
            checkpointing_options=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            eos_tokens=EOS_TOKENS,
            return_logprobs=True,
        ),
    )

    grpo_config = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
    )
    
    return cluster_config, grpo_config

def create_custom_optimizer(peak_lr: float = 3e-4, max_step: int = 100):
    update_schedule = config_for_function(cosine_with_linear_warmup).set(
        peak_lr=1.0,
        max_step=max_step,
        warmup_steps=50,
        begin_value=0.0,
        alpha=0.1,
    )
    
    base_optimizer_cfg = config_for_function(chain).set(
        args=[
            config_for_function(clip_by_global_norm).set(max_norm=1.0),
            config_for_function(adamw_decoupled_optimizer).set(
                learning_rate=peak_lr,
                b1=0.9,
                b2=0.95,
                eps=1e-8,
                update_schedule=update_schedule,
                weight_decay=0.1,
            ),
        ]
    )
    
    axlearn_opt = base_optimizer_cfg.instantiate()

    def init_fn(params):
        opt_params = jax.tree.map(
            lambda x: OptParam(value=x, factorization_spec=None, weight_decay_scale=1.0), params
        )
        return axlearn_opt.init(opt_params)

    def update_fn(updates, state, params=None):
        if params is not None:
            opt_params = jax.tree.map(
                lambda x: OptParam(value=x, factorization_spec=None, weight_decay_scale=1.0), params
            )
        else:
            opt_params = None
        return axlearn_opt.update(updates, state, opt_params)

    return optax.GradientTransformation(init_fn, update_fn)

class TunixConfigWrapper:
    def __init__(self, cfg):
        self._cfg = cfg
        self.num_layers = cfg.decoder.transformer.num_layers
        self.num_hidden_layers = cfg.decoder.transformer.num_layers
        self.hidden_size = cfg.decoder.dim
        self.vocab_size = cfg.decoder.vocab_size
        
        attn = cfg.decoder.transformer.layer.self_attention.attention
        self.num_attention_heads = attn.num_heads
        if hasattr(attn.input_linear, "num_kv_heads"):
            self.num_kv_heads = attn.input_linear.num_kv_heads
        elif hasattr(attn.input_linear, "input_linear") and hasattr(attn.input_linear.input_linear, "num_kv_heads"):
            self.num_kv_heads = attn.input_linear.input_linear.num_kv_heads
        else:
            self.num_kv_heads = attn.num_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
    def __getattr__(self, name):
        return getattr(self._cfg, name)

class AXLearnNNXWrapper(nnx.Module):
    def __init__(self, model_cfg, params):
        self.ax_model = model_cfg.set(name="model").instantiate(parent=None)
        self.config = TunixConfigWrapper(model_cfg)
        self.params = nnx.data(jax.tree.map(nnx.Param, params))

    def __call__(self, *args, **kwargs):
        input_ids = args[0] if len(args) > 0 else kwargs.get("input_batch", kwargs.get("input_ids"))
        cache = args[2] if len(args) > 2 else kwargs.get("cache")
        is_training = kwargs.get("is_training", False)

        if not isinstance(input_ids, dict):
            input_batch = {"input_ids": input_ids}
        else:
            input_batch = input_ids
            
        state = jax.tree.map(
            lambda p: getattr(p, "_raw_value", getattr(p, "value", p)) if isinstance(p, nnx.Variable) else p,
            self.params,
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )

        from axlearn.common.module import _global_context_stack
        if getattr(_global_context_stack, "thread_id", None) != threading.get_ident():
            install_context_stack([])

        print("Passed context stack")  
        (_, aux), _ = F(
            self.ax_model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(42),
            state=state,
            inputs=dict(input_batch=input_batch, return_aux=True),
        )
        print("passed Functional module")
        outputs = aux["logits"]
        print("passed output logits")
        outputs = jax.lax.with_sharding_constraint(outputs, jax.sharding.PartitionSpec())
        print("passed outputs sharding constraint")
        if len(args) >= 4 or "prefill" in kwargs or "cache" in kwargs:
            return outputs, cache
        return outputs


import numpy as np

class CallableInt(int):
    def __call__(self):
        return self

import numpy as np

class CallableInt(int):
    def __call__(self):
        return self

class TunixTokenizerWrapper:
    def __init__(self, tokenizer, max_prompt_length=512):
        self._tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        # DO NOT assign pad_id/eos_id/bos_id as attributes here.
        # Define them as methods below.

    # These methods satisfy the Tunix call signature: tokenizer.pad_id()
    def pad_id(self): return 0
    def eos_id(self): return 1
    def bos_id(self): return 0

    def pad_token_id(self): return 0
    def eos_token_id(self): return 1
    def bos_token_id(self): return 0

    def _pad_and_flatten(self, sequence):
        """Forces any input into a flat, 1D list of integers."""
        # Recursive flattening to handle JAX/TF tensors
        flat = []
        def _recurse(item):
            if hasattr(item, "numpy"): item = item.numpy()
            if hasattr(item, "tolist"):
                try: item = item.tolist()
                except: pass
            
            if isinstance(item, (list, tuple, np.ndarray)):
                for x in item: _recurse(x)
            else:
                try: flat.append(int(item))
                except: pass
        _recurse(sequence)
        
        # Leave room for BOS/EOS tokens added by Tunix so the padded length stays <= max_prompt_length
        target_len = max(1, self.max_prompt_length - 16)
        res = flat[:target_len]
        pad_len = target_len - len(res)
        if pad_len > 0:
            res.extend([self.pad_id()] * pad_len)
        return res

    def encode(self, text, *args, **kwargs):
        # 1. Clean input
        if hasattr(text, "numpy"): text = text.numpy()
        
        # 2. String path
        if isinstance(text, (str, bytes)):
            if isinstance(text, bytes): text = text.decode('utf-8', errors='ignore')
            try: enc = self._tokenizer.encode(text)
            except: enc = []
            return self._pad_and_flatten(enc)
            
        # 3. List/Array path (Assume already tokens)
        return self._pad_and_flatten(text)

    def __call__(self, text, *args, **kwargs):
        return self.encode(text)

    def tokenize(self, text, *args, **kwargs):
        return self.encode(text)

    def decode(self, ids, *args, **kwargs):
        flat = []
        def _flat(i):
            if isinstance(i, (list, tuple, np.ndarray)):
                for x in i: _flat(x)
            else:
                try: flat.append(int(i))
                except: pass
        _flat(ids)
        return self._tokenizer.decode(flat)

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if isinstance(conversation, str):
            text = conversation
        elif isinstance(conversation, list):
            text = "\n".join([msg.get("content", "") for msg in conversation if isinstance(msg, dict)])
        else:
            text = str(conversation)
            
        if tokenize:
            return self.encode(text)
        return text

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def prepare_math_dataset(batch_size: int, dataset_name="nvidia/OpenMathInstruct-1", test_size=0.01):
    """Loads the dataset in streaming mode and converts to tf.data using a generator."""
    print(f"Loading {dataset_name} dataset (streaming mode)...")
    
    # 1. Load in streaming mode
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True, streaming=True)
    
    # 2. Shuffle (buffer size needed for streaming)
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    
    # 3. Create split
    # Note: For streaming, we take/skip rather than train_test_split
    test_ds = dataset.take(1000)
    train_ds = dataset.skip(1000)

    # 4. Define formatting
    def format_example(example):
        return {
            "prompts": f"Question: {example.get('question', '')}\nAnswer: ",
            "ground_truth": example.get('expected_answer', '')
        }

    train_ds = train_ds.map(format_example)
    test_ds = test_ds.map(format_example)

    # 5. Manual conversion using from_generator
    def to_tf_dataset(hf_dataset, is_training):
        def generator():
            for row in hf_dataset:
                yield {
                    "prompts": row["prompts"],
                    "ground_truth": row["ground_truth"]
                }
        
        output_signature = {
            "prompts": tf.TensorSpec(shape=(), dtype=tf.string),
            "ground_truth": tf.TensorSpec(shape=(), dtype=tf.string),
        }
        
        tf_ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        
        if is_training:
            tf_ds = tf_ds.shuffle(1000)
        
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        return tf_ds.prefetch(tf.data.AUTOTUNE)

    return to_tf_dataset(train_ds, is_training=True), to_tf_dataset(test_ds, is_training=False)

def evaluate_model(rl_cluster, dataset, rollout_config, num_batches=1):
    print(f"--- Starting Evaluation ({num_batches} batches) ---")
    total_correct = 0
    total_samples = 0
    
    for i, batch in enumerate(dataset.take(num_batches)):
        prompts = batch["prompts"]
        if hasattr(prompts, "numpy"):
            prompts = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in prompts.numpy()]
            
        ground_truth = batch["ground_truth"]
        if hasattr(ground_truth, "numpy"):
            ground_truth = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in ground_truth.numpy()]
        
        rollout_output = rl_cluster.generate(prompts, rollout_config)
        
        if hasattr(rollout_output, "text"):
            completions = rollout_output.text
        elif hasattr(rollout_output, "completions"):
            completions = rollout_output.completions
        else:
            completions = rollout_output
            
        flat_completions = []
        for c in completions:
            if isinstance(c, (list, tuple, np.ndarray)):
                flat_completions.append(c[0])
            else:
                flat_completions.append(c)
        
        rewards = math_reward_fn(prompts, flat_completions, ground_truth=ground_truth)
        
        batch_correct = sum(rewards)
        total_correct += batch_correct
        total_samples += len(rewards)
        print(f"  Eval Batch {i+1}: Accuracy = {batch_correct / max(1, len(rewards)):.4f}")
        
    accuracy = total_correct / max(1, total_samples)
    print(f"--- Final Evaluation Accuracy: {accuracy:.4f} ---")
    return accuracy

def main(_):
    #if jax.process_index() == 0:
    wandb.init()
    global_batch_size = jax.process_count() * 2
    train_ds, test_ds = prepare_math_dataset(batch_size=global_batch_size)

    print("Successfully loaded training dataset:", train_ds)
    print("Training batch features:", next(iter(train_ds)).keys())

    print("\nSuccessfully loaded test dataset:", test_ds)
    print("Test batch features:", next(iter(test_ds)).keys())

    trainer_cfg, restored_state, mesh = get_fuji_trainer_and_state(FLAGS.ckpt_dir)

    optimizer = create_custom_optimizer()

    import tunix.generate.sampler as sampler_module
    if not hasattr(sampler_module.Sampler, "_patched_for_replication"):
        _orig_init = sampler_module.Sampler.__init__

        @jax.jit
        def _gather_array(x):
            return jax.lax.with_sharding_constraint(x, jax.sharding.PartitionSpec())

        def _force_replicate_state(state):
            def _replicate(x):
                if isinstance(x, jax.Array) and hasattr(x, "ndim") and x.ndim <= 3:
                    if not (getattr(x, "is_fully_replicated", False) or getattr(x, "is_fully_addressable", False)):
                        try:
                            return _gather_array(x)
                        except Exception:
                            pass
                return x
            return jax.tree_util.tree_map(_replicate, state)

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            orig_prefill = self._compiled_prefill_fn
            orig_decode = self._compiled_decode_fn
            
            def prefill_wrapper(*p_args, **p_kwargs):
                return _force_replicate_state(orig_prefill(*p_args, **p_kwargs))
            
            def decode_wrapper(*d_args, **d_kwargs):
                return _force_replicate_state(orig_decode(*d_args, **d_kwargs))
                
            self._compiled_prefill_fn = prefill_wrapper
            self._compiled_decode_fn = decode_wrapper

        sampler_module.Sampler.__init__ = _patched_init
        sampler_module.Sampler._patched_for_replication = True

    cluster_config, grpo_config = create_tunix_config(
        mesh, optimizer, FLAGS.ckpt_dir
    )
    print("Successfully created Tunix configuration.")

    print("Instantiating GRPO Learner...")
    base_tokenizer = config_for_function(vocab).set(sentencepiece_model_name="bpe_32k_c4.model").instantiate()
    tokenizer = TunixTokenizerWrapper(base_tokenizer)
    base_model_cfg = trainer_cfg.model
    
    actor_nnx = AXLearnNNXWrapper(base_model_cfg, restored_state.model)
    reference_nnx = AXLearnNNXWrapper(base_model_cfg, restored_state.model)
    
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_nnx,
        reference=reference_nnx,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[math_reward_fn],
        algo_config=grpo_config,
    )
    
    print("GRPO Learner instantiated successfully!")

    print("Running Pre-Training Evaluation...")
    evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config, num_batches=1)

    print("Starting GRPO training...")
    grpo_trainer.train(train_ds, test_ds)

    print("Running Post-Training Evaluation...")
    evaluate_model(rl_cluster, test_ds, cluster_config.rollout_config, num_batches=2)
    #if jax.process_index() == 0:
    wandb.finish()

if __name__ == "__main__":
    #jax.distributed.initialize()
    pathwaysutils.initialize()
    print(len(jax.devices()))
    app.run(main)