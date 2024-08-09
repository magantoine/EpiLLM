# python requirements
# FROM --platform=linux/amd64 python:3.10
FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.07-py3

# # Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN pip install --upgrade pip

RUN apt-get update
RUN apt-get install nano


RUN --mount=type=secret,id=my_env,dst=/tmp/my_env cat /tmp/my_env > /tmp/envfile

ARG USER_NAME=magron
ARG USER_ID=189416

ARG GROUP_NAME=NLP-StaffU
ARG GROUP_ID=11131

RUN useradd -m -u $USER_ID $USER_NAME
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN usermod -g ${GROUP_NAME} ${USER_NAME}
# USER ${USER_NAME}


# # Define container entry point (could also work with CMD python main.py)


### PUTS EVERYTHING TO home/magron dir

### CONDA :
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
# RUN arch=$(uname -m) && \
#     if [ "$arch" = "x86_64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
#     elif [ "$arch" = "aarch64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
#     else \
#     echo "Unsupported architecture: $arch"; \
#     exit 1; \
#     fi && \
#     wget $MINICONDA_URL -O miniconda.sh && \
#     mkdir -p /root/.conda && \
#     bash miniconda.sh -b -p /root/miniconda3 && \
#     rm -f miniconda.sh

# RUN conda --version


COPY . /home/magron/

RUN touch /home/magron/.env
RUN echo "DIR_PATH=\"home/magron/\"" > /home/magron/.env
RUN echo "API_KEY=\"1\"" >> /home/magron/.env
RUN echo "DEVICE=\"cuda\"" >> /home/magron/.env
RUN echo "OS_TYPE=\"Darwin\"" >> /home/magron/.env

USER ${USER_NAME}

# RUN pip install -r /home/magron/requirements.txt --no-cache-dir
# RUN pip install huggingface_hub --no-cache-dir
# RUN pip install pathlib --no-cache-dir
# RUN pip install accelerate -U






## set home dir to working directory
WORKDIR /home/magron/

# RUN echo "./prepare_container.sh && pip install wandb && python training.local.py --datasets pmc pubmed --save_dir scratch/home/magron --checkpoint epitron_PMC_FullPubmed --base_checkpoint meta-llama/Meta-Llama-3-8B" > run.sh
# RUN echo "ls -l; pwd; ls -l /" >> run.sh


# ENTRYPOINT ["bash", "run.sh"]
ENTRYPOINT ["python", "container_link.py"]

## args : 
#   - save_dir : directory to save the checkpoints, scratch_dir in HaaS001 is at scratch/home/magron, we add the checkpoints to have them all in one point
#   - checkpoint : epitron_tv0
#   - dataset : PMC (for tv0)
#   - base_checkpoint : LLaMa3
#   - batch_size : 16
#   - n_train_epoch : 1

# ENTRYPOINT ["python", "training.py","--save_dir", "scratch/home/magron/checkpoints","--checkpoint", "epitron_tv0", "--datasets", "pmc", "--base_checkpoint", "meta-llama/Meta-Llama-3-8B","--batch_size", "2","--n_train_epoch", "1"]


# ENTRYPOINT ["python", "training.py", "--type", "cmd"]