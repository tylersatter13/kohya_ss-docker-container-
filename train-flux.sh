#! /bin/bash

set -x

# set up env
export HF_HOME="${HF_HOME:-/workspace/hf}"

# set up hyperparams
PARAMS_D_COEF="${PARAMS_D_COEF:-2}"
PARAMS_LEARNING_RATE="${PARAMS_LEARNING_RATE:-1}"
PARAMS_LEARNING_SCHEDULER="${PARAMS_LEARNING_SCHEDULER:-constant}"
PARAMS_MAX_EPOCHS="${PARAMS_MAX_EPOCHS:-5}"
PARAMS_NETWORK_ALPHA="${PARAMS_NETWORK_ALPHA:-4}"
PARAMS_NETWORK_DIM="${PARAMS_NETWORK_DIM:-32}"
PARAMS_NOISE_OFFSET="${PARAMS_NOISE_OFFSET:-0.1}"
PARAMS_OPTIMIZER_ARGS="${PARAMS_OPTIMIZER_ARGS:-}"
PARAMS_OPTIMIZER_TYPE="${PARAMS_OPTIMIZER_TYPE:-prodigy}"
PARAMS_RESUME_NULL="${PARAMS_RESUME_NULL:-}"
PARAMS_SAVE_EVERY_N_STEPS="${PARAMS_SAVE_EVERY_N_STEPS:-100}"
PARAMS_SAMPLE_EVERY_N_STEPS="${PARAMS_SAMPLE_EVERY_N_STEPS:-200}"
PARAMS_SPLIT_QKV="${PARAMS_SPLIT_QKV:-False}"
PARAMS_WEIGHT_DECAY="${PARAMS_WEIGHT_DECAY:-0.1}"

# derived params
DEFAULT_NUM_CYCLES="$(( ${PARAMS_MAX_EPOCHS} / 3 ))"
PARAMS_NUM_CYCLES="${PARAMS_NUM_CYCLES:-${DEFAULT_NUM_CYCLES}}"

echo "Training Flux.1 Dev for ${PARAMS_MAX_EPOCHS} epochs using the ${PARAMS_OPTIMIZER_TYPE} optimizer and ${PARAMS_LEARNING_SCHEDULER} scheduler..."
time accelerate launch \
    --dynamo_backend no \
    --mixed_precision bf16 \
    --num_cpu_threads_per_process 2 \
    --num_machines 1 \
    --num_processes 0 \
    /opt/sd-scripts/flux_train_network.py \
    --pretrained_model_name_or_path /models/flux_dev.safetensors \
    --clip_l /models/clip_l.safetensors \
    --t5xxl /models/t5xxl_fp16.safetensors \
    --ae /models/flux_ae.safetensors \
    --cache_latents_to_disk \
    --save_model_as safetensors \
    --sdpa \
    --persistent_data_loader_workers \
    --max_data_loader_n_workers 4 \
    --seed 42 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --save_precision bf16 \
    --network_module networks.lora_flux \
    --network_args "train_t5xxl=False" "split_qkv=${PARAMS_SPLIT_QKV}" \
    --network_alpha ${PARAMS_NETWORK_ALPHA} \
    --network_dim ${PARAMS_NETWORK_DIM} \
    --optimizer_type ${PARAMS_OPTIMIZER_TYPE} \
    --optimizer_args \
        "decouple=True" \
        "weight_decay=${PARAMS_WEIGHT_DECAY}" \
        "betas=0.9,0.999" \
        "use_bias_correction=False" \
        "safeguard_warmup=False" \
        "d_coef=${PARAMS_D_COEF}" \
    --cache_text_encoder_outputs \
    --cache_text_encoder_outputs_to_disk \
    --lr_scheduler "${PARAMS_LEARNING_SCHEDULER}" \
    --lr_scheduler_num_cycles "${PARAMS_NUM_CYCLES}" \
    --noise_offset "${PARAMS_NOISE_OFFSET}" \
    --learning_rate "${PARAMS_LEARNING_RATE}" \
    --fp8_base \
    --highvram \
    --max_train_epochs "${PARAMS_MAX_EPOCHS}" \
    --save_every_n_steps "${PARAMS_SAVE_EVERY_N_STEPS}" \
    --sample_every_n_steps "${PARAMS_SAMPLE_EVERY_N_STEPS}" \
    --sample_prompts "${SAMPLE_PROMPTS}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    --timestep_sampling "${PARAMS_TIMESTEP_SAMPLING:-sigmoid}" \
    --model_prediction_type raw \
    --discrete_flow_shift 3.1582 \
    --guidance_scale 1.0 \
    --loss_type l2

echo "Training complete."
sleep 30m
