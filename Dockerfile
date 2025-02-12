FROM rocm/pytorch:rocm6.3.2_ubuntu24.04_py3.12_pytorch_release_2.4.0

# download Flux.1 Dev FP8
RUN mkdir /models \
 && wget --progress=bar:force:noscroll \
    -O /models/flux1-dev-fp8.safetensors \
    https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors

# download sd-scripts (requirements are/were missing torchvision and W&B)
RUN git clone \
    --single-branch --branch sd3 \
    https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts

# set up venv
RUN cd /opt/sd-scripts \
 && pip3 install -r requirements.txt

# fixes for boto
RUN apt remove python3-botocore \
 && pip3 uninstall -y botocore \
 && apt install -y python3-botocore \
 && pip3 install --upgrade boto3 

# set up training scripts
COPY train-*.sh /opt/sd-scripts/
COPY dataset.json /opt/sd-scripts/

# configure to download dataset and run training
ENTRYPOINT [ "/opt/sd-scripts/train-main.sh" ]