set -euxo pipefail

# -e DATASET_NAME="blondedman/the-criterion-collection" \


sudo docker build -t onboarding .
sudo docker run \
    -it \
    --rm \
    -e BASE_MODEL=flux \
    -e DATASET_NAME="amitkumargurjar/car-detection-and-tracking-dataset" \
    -e DATASET_CONFIG="/opt/sd-scripts/dataset.json" \
    -e SAMPLE_PROMPTS="/opt/sd-scripts/prompts.txt" \
    onboarding
