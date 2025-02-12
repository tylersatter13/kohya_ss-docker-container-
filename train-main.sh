#! /bin/bash

DATASET_HOST="${DATASET_HOST:-https://www.kaggle.com/api/v1/datasets/download}"
DATASET_NAME="${DATASET_NAME:-}"
DATASET_PATH="${DATASET_PATH:-/workspace/dataset}"

if [[ -z "${DATASET_NAME}" ]];
then
    echo "DATASET_NAME is required"
    exit 1
fi

echo "Downloading dataset to: ${DATASET_PATH}"
mkdir -p "${DATASET_PATH}" || true
curl -L -o "${DATASET_PATH}/dataset.zip" \
    "${DATASET_HOST}/${DATASET_NAME}"
unzip "${DATASET_PATH}/dataset.zip"

BASE_MODEL="${BASE_MODEL:-flux}"
echo "Running ${BASE_MODEL} training..."
/opt/sd-scripts/train-${BASE_MODEL}.sh 
