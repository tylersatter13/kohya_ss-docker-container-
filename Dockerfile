FROM rocm/pytorch:rocm6.3.2_ubuntu24.04_py3.12_pytorch_release_2.4.0

# Create models directory
RUN mkdir /models

# download sd-scripts (requirements are/were missing torchvision and W&B)
RUN git clone \
    --single-branch --branch sd3 \
    https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts

# set up venv
RUN cd /opt/sd-scripts \
 && pip3 install -r requirements.txt

# Copy startup script
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Optionally set as entrypoint
# ENTRYPOINT ["/usr/local/bin/startup.sh"]
# Or instruct users to run manually