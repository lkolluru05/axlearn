from orbax.checkpoint import PyTreeCheckpointHandler, CheckpointManager, Checkpointer

ckpt_dir="gs://tpu-prod-env-multipod-axlearn/lkolluru-apple-orbax/orbax-ckp/checkpoints/tensors"



handler = PyTreeCheckpointHandler()
checkpointer = Checkpointer(handler)  # ✅ Wrap handler

manager = CheckpointManager(ckpt_dir, {'state': checkpointer})