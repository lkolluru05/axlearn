import tensorflow as tf
from absl import app, flags
from axlearn.common import input_tf_data
from axlearn.common.config import config_for_class, config_for_function
import jax
import pathwaysutils
import jax.numpy as jnp
from jax.sharding import Mesh
from axlearn.common.checkpointer import Checkpointer
from axlearn.common.trainer import TrainerState
from axlearn.common.base_layer import ParameterSpec
from axlearn.common.utils import PartitionSpec, create_device_mesh
from axlearn.experiments.text.gpt.c4_trainer import named_trainer_configs
from axlearn.experiments.text.common import vocab
from axlearn.common.learner import Learner
from axlearn.common.optimizers import adamw_decoupled_optimizer, chain, clip_by_global_norm
from axlearn.common.schedule import cosine_with_linear_warmup
from axlearn.common.update_transformation import ConditionalUpdateTransformation

# Tunix imports (adjust the paths as necessary for your environment)
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "gs://axlearn-public/tensorflow_datasets", "Directory containing the TFDS dataset.")
flags.DEFINE_string("ckpt_dir", "gs://tpu-prod-env-multipod-axlearn/elastic-test30/checkpoints", "Directory containing the checkpoints.")

def get_dataset_config(is_training: bool, dataset_name: str, split: str, batch_size: int = 32, data_dir: str = None):
    """Constructs an AXLearn dataset pipeline config.
    
    Args:
        is_training: Whether the dataset is used for training.
        dataset_name: The TFDS dataset name (e.g., 'mnist', 'c4/en:3.0.1').
        split: The TFDS dataset split (e.g., 'train', 'test').
        batch_size: Global batch size.
        data_dir: Path to the TFDS data directory.
    """
    # 1. Define the read config
    read_config = config_for_function(input_tf_data.tfds_read_config)

    # 2. Define the source
    source_cfg = config_for_function(input_tf_data.tfds_dataset).set(
        dataset_name=dataset_name,
        split=split,
        is_training=is_training,
        read_config=read_config,
        data_dir=data_dir,
        train_shuffle_buffer_size=1024 * 16,
    )

    # 3. Define the processor (using identity to yield raw output without modification)
    processor_cfg = config_for_function(input_tf_data.identity)

    # 4. Define the batcher
    batcher_cfg = config_for_function(input_tf_data.batch).set(
        global_batch_size=batch_size,
        pad_example_fn=input_tf_data.default_pad_example_fn,
    )

    # 5. Construct the overall Input config
    return input_tf_data.Input.default_config().set(
        name="input",
        is_training=is_training,
        source=source_cfg,
        processor=processor_cfg,
        batcher=batcher_cfg,
    )

def get_dataset(is_training: bool, dataset_name: str, split: str, batch_size: int = 32, data_dir: str = None):
    """Instantiates an AXLearn dataset pipeline."""
    input_cfg = get_dataset_config(is_training, dataset_name, split, batch_size, data_dir)

    # Instantiate the input module and get the underlying tf.data.Dataset
    input_module = input_cfg.instantiate(parent=None)
    return input_module.dataset()

def get_fuji_trainer_and_state(ckpt_dir: str):
    """Gets a Fuji trainer config and restores its state from a checkpoint."""
    # 1. Get the Fuji trainer config
    trainer_cfg = named_trainer_configs()["fuji-7B-v2-flash"]()
    model_cfg = trainer_cfg.model
    print("Successfully loaded Fuji trainer config.")

    # 3. Instantiate the checkpointer
    checkpointer = Checkpointer.default_config().set(name="checkpointer", dir=ckpt_dir).instantiate(parent=None)

    # 4. Restore the state within the appropriate Mesh context
    mesh = Mesh(create_device_mesh(mesh_shape=trainer_cfg.mesh_shape), trainer_cfg.mesh_axis_names)
    with mesh:
        # 2. Define the expected trainer state structure
        model = model_cfg.set(name="model").instantiate(parent=None)
        model_param_specs = model.create_parameter_specs_recursively()
        
        learner = trainer_cfg.learner.set(name="learner").instantiate(parent=None)
        learner_state_specs = learner.create_state_partition_specs(model_param_specs)

        trainer_state_specs = TrainerState(
            prng_key=ParameterSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
            model=model_param_specs,
            learner=learner_state_specs,
        )

        step, restored_state_dict = checkpointer.restore(step=None, state=trainer_state_specs._asdict())
        # Convert the dictionary back to TrainerState
        restored_state = TrainerState(**{k: v for k, v in restored_state_dict.items() if k in TrainerState._fields})
        
    print(f"Restored state from step: {step}")
    return trainer_cfg, restored_state, mesh

def dummy_reward_fn(generations, **kwargs):
    """A dummy reward function returning 1.0 for each generation."""
    return jnp.ones(generations.shape[:2], dtype=jnp.float32)

def create_tunix_config(mesh, optimizer, ckpt_dir: str):
    """Creates the Tunix cluster and GRPO configurations."""
    # Define configuration constants
    EVAL_EVERY_N_STEPS = 100
    MAX_STEPS = 1000
    TRAIN_MICRO_BATCH_SIZE = 8
    TOTAL_GENERATION_STEPS = 256
    MAX_PROMPT_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.95
    TOP_K = 50
    EOS_TOKENS = [128001]
    NUM_GENERATIONS = 4
    NUM_ITERATIONS = 1
    BETA = 0.01
    EPSILON = 0.2

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
            # metrics logging
            metrics_logging_options=None,
            # checkpoint saving
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
    """Creates a custom optimizer utilizing AXLearn's update transformations."""
    # 1. Define the learning rate schedule
    update_schedule = config_for_function(cosine_with_linear_warmup).set(
        peak_lr=1.0,
        max_step=max_step,
        warmup_steps=50,
        begin_value=0.0,
        alpha=0.1,
    )
    
    # 2. Define the base optimizer transformation (chaining clip and adamw)
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
    
    # 3. Use ConditionalUpdateTransformation to conditionally skip updates
    conditional_opt_cfg = ConditionalUpdateTransformation.default_config().set(
        inner=base_optimizer_cfg,
        update_schedule=lambda step: True,  # e.g. `lambda step: step % 2 == 0`
    )
    
    return conditional_opt_cfg

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from axlearn.common.checkpointer import Checkpointer
from axlearn.common.trainer import TrainerState

def extract_model_params_from_axlearn_checkpoint(ckpt_dir, trainer_mesh, dtype):
    """
    Loads an AXLearn checkpoint and shards the model parameters identically
    to params_lib.create_model_from_safe_tensors.
    """
    # 1. Instantiate a default checkpointer targeting your GCS or local directory
    checkpointer = Checkpointer.default_config().set(
        name="conversion_checkpointer", 
        dir=ckpt_dir
    ).instantiate(parent=None)

    with trainer_mesh:
        # 2. Restore the raw, un-sharded checkpoint state structure
        # Passing state=None allows the checkpointer to return the raw underlying dict
        step, restored_dict = checkpointer.restore(step=None, state=None)
        print(f"Extracted AXLearn checkpoint from step: {step}")
        
        # 3. Grab ONLY the model weights (discarding 'learner', 'step', 'prng_key')
        raw_model_params = restored_dict["model"]
        
        # 4. Cast and Shard the variables over your live trainer_mesh context
        # This mirrors what create_model_from_safe_tensors does internally.
        def shard_and_cast_tensor(leaf):
            # Cast the array to your desired precision (e.g., jnp.bfloat16)
            casted_leaf = leaf.astype(dtype)
            
            # Build an un-sharded/replicated sharding specification across the active mesh
            # If your model needs specific axis-sharding rules, replace P() with your target PartitionSpec
            sharding = NamedSharding(trainer_mesh, P()) 
            
            # Distribute the array block across the active TPU cluster
            return jax.device_put(casted_leaf, sharding)
            
        # Recursively apply casting and mesh placement across the nested dictionary tree
        sharded_model_params = jax.tree_util.tree_map(
            shard_and_cast_tensor, 
            raw_model_params
        )
        
        return sharded_model_params

def main(_):
    train_ds = get_dataset(
        is_training=True, dataset_name="c4/en:3.0.1", split="train", batch_size=8, data_dir=FLAGS.data_dir
    )
    test_ds = get_dataset(
        is_training=False, dataset_name="c4/en:3.0.1", split="validation", batch_size=8, data_dir=FLAGS.data_dir
    )

    print("Successfully loaded training dataset:", train_ds)
    print("Training batch features:", next(iter(train_ds)).keys())

    print("\nSuccessfully loaded test dataset:", test_ds)
    print("Test batch features:", next(iter(test_ds)).keys())

    trainer_cfg, restored_state, mesh = get_fuji_trainer_and_state(FLAGS.ckpt_dir)
    trainer_mesh = Mesh(
        create_device_mesh(mesh_shape=trainer_cfg.mesh_shape), 
        trainer_cfg.mesh_axis_names
    )
    #shared_params_model = extract_model_params_from_axlearn_checkpoint(FLAGS.ckpt_dir, trainer_mesh, jnp.bfloat16)


    optimizer = create_custom_optimizer()
    
    cluster_config, grpo_config = create_tunix_config(
        mesh, optimizer, FLAGS.ckpt_dir
    )
    print("Successfully created Tunix configuration.")

    print("Instantiating GRPO Learner...")
    tokenizer = config_for_function(vocab).set(sentencepiece_model_name="bpe_32k_c4.model").instantiate()
    
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=trainer_cfg,
        reference=trainer_cfg,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    rl_cluster.sync_weights(actor_weights=restored_state.model)

    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[dummy_reward_fn],
        algo_config=grpo_config,
    )
    
    print("GRPO Learner instantiated successfully!")

    print("Starting GRPO training...")
    # The restored_state contains your pre-trained model weights. 
    # You pass it to the train method so the actor and reference models 
    # start from the checkpoint rather than from random initialization.
    grpo_trainer.train(train_ds, test_ds)

if __name__ == "__main__":
    pathwaysutils.initialize()
    print(len(jax.devices()))
    app.run(main)
