set -euxo pipefail

IMAGE_NAME="${IMAGE_NAME:-ssuberunpod/runpod-onboarding:latest}"

docker build -t "${IMAGE_NAME}" .
docker push "${IMAGE_NAME}"