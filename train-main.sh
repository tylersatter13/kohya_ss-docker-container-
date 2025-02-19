#! /bin/bash

export DATASET_CONFIG="${DATASET_CONFIG:-}"
export DATASET_HOST="${DATASET_HOST:-https://www.kaggle.com/api/v1/datasets/download}"
export DATASET_NAME="${DATASET_NAME:-}"
export DATASET_PATH="${DATASET_PATH:-/workspace/dataset}"
export OUTPUT_PATH="${OUTPUT_PATH:-/workspace/output}"
export SAMPLE_PROMPTS="${SAMPLE_PROMPTS:-}"

if [[ -z "${DATASET_CONFIG}" ]];
then
    echo "DATASET_CONFIG is required"
    exit 1
fi

if [[ -z "${DATASET_NAME}" ]];
then
    echo "DATASET_NAME is required"
    exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]];
then
    echo "OUTPUT_PATH is required"
    exit 1
fi

if [[ -z "${SAMPLE_PROMPTS}" ]];
then
    echo "SAMPLE_PROMPTS is required"
    exit 1
fi

mkdir -p "${OUTPUT_PATH}" || true

echo "Downloading dataset to: ${DATASET_PATH}"
mkdir -p "${DATASET_PATH}" || true

if [[ ! -f "${DATASET_PATH}/dataset.zip" ]];
then
    curl -L -o "${DATASET_PATH}/dataset.zip" \
        "${DATASET_HOST}/${DATASET_NAME}"
    (cd "${DATASET_PATH}" && unzip "${DATASET_PATH}/dataset.zip")
fi

export BASE_MODEL="${BASE_MODEL:-flux}"
echo "Running ${BASE_MODEL} training..."
/opt/sd-scripts/train-${BASE_MODEL}.sh 
