# Training Flux.1 Dev on the MI300x GPU with Huge Batch Sizes

For this post, I'm going to be experimenting with fine-tuning Flux.1 Dev on the
world's largest GPU, the Mi300x. At 192GB of VRAM per GPU, the Mi300x supports
training at batch sizes and resolutions that no other card can handle.

## Prerequisites

To run this yourself, you will need:

1. a RunPod account
2. a text editor
3. one of:
    - a local installation of Docker and a Docker Hub account
    - OR Github actions and a Github account

You do not need to run anything locally to complete this, but it may be easier to build the
Docker container on a local machine with plenty of free disk space. Pushing the image up to
GHCR or Docker Hub may be faster from Github Actions, but you may need to remove some unused
dependencies and build tools in order to free up enough disk space. 

If you have limited bandwidth or a slow network connection, I recommend using Github Actions,
since the Flux models can be pretty large.

## Container

In order to keep all of the dependencies strongly versioned and portable, we will build a
Docker container with everything you need to train Flux.1 on AMD GPUs.

The Dockerfile will do a few different things:

1. Start with the `rocm/pytorch` base image
2. Download the Flux.1 models
3. Install the kohya-ss/sd-scripts repo for training
4. Fix a few conflicting dependencies
5. Set up the training scripts

The only thing that will not be baked into the Docker container for this example are the
training dataset and hyperparameters used to configure the training run.

### Base Container

To avoid building PyTorch for each version of ROCm, we will be using the `rocm/pytorch`
base image. This is another instance where we will be trading disk space for time: the
ROCm images are quite large, but building PyTorch can take a long time, especially if
you have to negotiate with different driver versions on your local and remote machines.

Setting up ROCm to build PyTorch from scratch is outside of the scope of this tutorial.

### Flux Model

To keep things simple, I am downloading the Flux.1 Dev model from a Cloudflare mirror.
You can download Flux from HuggingFace or another source, as long as they follow the
format required by the sd-scripts repository, with the CLIP-L and T5XXL encoders in 
separate files.

This is the largest part of the container and something you will want to run once,
then reuse that part of the container as much as possible.

### Training Scripts

To make the training container more flexible, I opted to extract most of the 
hyperparameters into environment variables, making it easy to modify them for each
training run without rebuilding the image or changing the dataset configuration.

## Template

Once you have built and pushed the Docker container to Docker Hub or Github Container
Registry, we will create a RunPod template that will let you use that container for
training without configuring each of the individual parameters for every run.

Go into the RunPod console and create a new template:

TODO: screenshot

There are a few environment variables that need to be configured for the template:

TODO

## Training

Once you have configured a template, you should create a pod based on that template,
with 1 of the Mi300x GPUs attached.

It will take a little while for the container image to be downloaded and extracted,
but once that is ready, the scripts will start downloading the dataset and training
the Flux LoRA automatically. 

Once training is done, you can download the results using SSH or a file transfer tool
like rsync or SyncThing. Tools that support checksumming the files and only downloading
new/changed files are ideal here, because training will save a checkpoint every so often.
It's a good idea to download all of the checkpoints so that you can sort through them
offline and select the best option (or even merge a few, if none of them were quite right).
