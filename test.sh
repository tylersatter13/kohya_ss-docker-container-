set -euxo pipefail

sudo docker build -t onboarding .
sudo docker run \
    -it \
    --rm \
    -e BASE_MODEL=flux \
    -e DATASET_NAME="blondedman/the-criterion-collection" \
    onboarding
