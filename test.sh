set -euxo pipefail

sudo docker build -t onboarding .
sudo docker run \
    -it \
    --rm \
    -e BASE_MODEL=flux \
    -e DATASET_NAME="blondedman/the-criterion-collection" \
    -e DATASET_CONFIG="/opt/sd-scripts/dataset.json" \
    -e SAMPLE_PROMPTS="test" \
    onboarding
