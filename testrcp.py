import os
import time
import huggingface_hub

print(os.environ)
print(os.getuid())
print(os.popen('whoami').read())

with open("/tmp/envfile", 'r') as f:
    HF_TOKEN = f.read().split("=")[1][1:-1]

huggingface_hub.login(HF_TOKEN)

print("Hello World !!!")

print(os.listdir())
print(os.listdir("home/magron"))
print(os.listdir("home/magron/output"))
print(os.listdir("home/magron/output/home"))

