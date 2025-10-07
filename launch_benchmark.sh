
export BUCKET="cloud-tpu-multipod-dev-euw4"
export CHECKPOINT_DIR="gs://${BUCKET}/axlearn-fuji-v3-70b/checkpoints/step_00000100"
# Generate a random ID
export RANDOM_CHARS=$(LC_CTYPE=C openssl rand -base64 12 | tr -dc 'a-z0-9' | head -c 4 ; echo)
export PROFILE_DIR="gs://${BUCKET}/profiles/${RANDOM_CHARS}/"
# export RUNNER_NAME=gke_tpu_single
export RUNNER_NAME=gke_tpu_pathways
export JOBSET_NAME=${JOBSET_NAME:-$USER}-6

# check if environment variable JAX_PLATFORMS equals proxy
if [[ "$JAX_PLATFORMS" == "proxy" ]]; then
  echo "JAX_PLATFORMS is set to 'proxy'."
  TPU_PREMAPPED_BUFFER_SIZE=34359738368 \
           python3 test_benchmark.py --ckpt_path=${CHECKPOINT_DIR}
else

export CLUSTER=$(axlearn gcp config | grep gke_cluster | \
                 awk '{ print $3 }' | tr -d  '"')
  #--queue=multislice-queue \
axlearn gcp launch run --cluster=$CLUSTER \
        --runner_name ${RUNNER_NAME} \
        --name=${JOBSET_NAME} \
        --instance_type="tpu-v5p-32" \
        --reservation="cloudtpu-20240716121201-595617744" \
        --num_replicas=1 \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- TPU_PREMAPPED_BUFFER_SIZE=34359738368 \
           python3 test_benchmark.py --ckpt_path=${CHECKPOINT_DIR}
           # --profile_dir="${PROFILE_DIR}"


           # --use_colocated_python=false

fi
