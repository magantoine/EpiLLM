# python requirements
FROM --platform=linux/amd64 python:3.12.1
# # Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]


# FROM python:3.12.1

# Add requirements file in the container
# COPY requirements.txt ./requirements.txt

## generate THE .env file
# RUN mkdir epitron
# COPY . ./epitron
# RUN touch epitron/.env
# RUN echo "DIR_PATH=\"epitron/\"" > epitron/.env
# RUN echo "API_KEY=\"1\"" >> epitron/.env
# RUN echo "DEVICE=\"cuda\"" >> epitron/.env
# RUN echo "OS_TYPE=\"Darwin\"" >> epitron/.env
# RUN cat epitron/.env

# RUN pip install --upgrade pip
# RUN pip install -r epitron/requirements.txt

RUN mkdir /mnt/u14157_ic_nlp_001_files_nfs
RUN mkdir /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1



RUN --mount=type=secret,id=my_env,dst=/tmp/my_env cat /tmp/my_env > /tmp/envfile

ARG USER_NAME=magron
ARG USER_ID=189416

ARG GROUP_NAME=NLP-StaffU
ARG GROUP_ID=11131


# # Define container entry point (could also work with CMD python main.py)
# ENTRYPOINT ["python", "epitron/training.py","--save_dir", "checkpoints","--checkpoint", "epitron_tv0", "--datasets", "pmc", "--base_checkpoints", "meta-llama/Meta-Llama-3-8B","--batch_size", "4","--n_train_epoch", "1"]
RUN useradd -m -u $USER_ID $USER_NAME
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}

RUN chown magron /mnt/u14157_ic_nlp_001_files_nfs

USER ${USER_NAME}


RUN mkdir /home/magron/output


## prepare to mount







RUN pip install huggingface_hub

COPY testrcp.py ./testrcp.py



ENTRYPOINT ["python", "testrcp.py"]