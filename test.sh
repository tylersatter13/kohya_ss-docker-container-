set -euxo pipefail

IMAGE_NAME="${IMAGE_NAME:-ssuberunpod/runpod-onboarding:latest}"

docker build -t "${IMAGE_NAME}" .
docker run \
    -it \
    --rm \
    -e BASE_MODEL=flux \
    -e DATASET_NAME="amitkumargurjar/car-detection-and-tracking-dataset" \
    -e DATASET_CONFIG="/opt/sd-scripts/dataset.json" \
    -e SAMPLE_PROMPTS="/opt/sd-scripts/prompts.txt" \
    "${IMAGE_NAME}"
