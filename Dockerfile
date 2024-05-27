# python requirements
FROM --platform=linux/amd64 python:3.12.1
# # Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN pip install --upgrade pip


RUN --mount=type=secret,id=my_env,dst=/tmp/my_env cat /tmp/my_env > /tmp/envfile

ARG USER_NAME=magron
ARG USER_ID=189416

ARG GROUP_NAME=NLP-StaffU
ARG GROUP_ID=11131

RUN useradd -m -u $USER_ID $USER_NAME
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN usermod -g ${GROUP_NAME} ${USER_NAME}
USER ${USER_NAME}


# # Define container entry point (could also work with CMD python main.py)


### PUTS EVERYTHING TO home/magron dir
COPY . ./home/magron/
RUN touch home/magron/.env
RUN echo "DIR_PATH=\"home/magron/\"" > /home/magron/.env
RUN echo "API_KEY=\"1\"" >> /home/magron/.env
RUN echo "DEVICE=\"cuda\"" >> /home/magron/.env
RUN echo "OS_TYPE=\"Darwin\"" >> /home/magron/.env
RUN cat /home/magron/.env

RUN pip install -r /home/magron/requirements.txt --no-cache-dir
RUN pip install huggingface_hub --no-cache-dir
RUN pip install pathlib --no-cache-dir

## set home dir to working directory
WORKDIR /home/magron/



# ENTRYPOINT ["python", "testrcp.py"]

## args : 
#   - save_dir : directory to save the checkpoints, scratch_dir in HaaS001 is at scratch/home/magron, we add the checkpoints to have them all in one point
#   - checkpoint : epitron_tv0
#   - dataset : PMC (for tv0)
#   - base_checkpoint : LLaMa3
#   - batch_size : 16
#   - n_train_epoch : 1

ENTRYPOINT ["python", "training.py","--save_dir", "scratch/home/magron/checkpoints","--checkpoint", "epitron_tv0", "--datasets", "pmc", "--base_checkpoint", "meta-llama/Meta-Llama-3-8B","--batch_size", "4","--n_train_epoch", "1"]