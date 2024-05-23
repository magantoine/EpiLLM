### (1) BUILD THE DOCKER IMAGE
V=`cat .run_id`
echo "tv0-$V"

docker build -f Dockerfile \
  -t ic-registry.epfl.ch/nlp/magron/nlp-magron:tv0-$V . \
  --secret id=my_env,src=.docker.env


# ### (2) PUSH IT TO REMOTE
docker push ic-registry.epfl.ch/nlp/magron/nlp-magron:tv0-$V


# ## (3) SUBMIT TO RUNAI 
runai submit \
  --name epitron-train-tv0-$V \
  -i ic-registry.epfl.ch/nlp/magron/nlp-magron:tv0-$V \
  --gpu 1 --cpu 1 \
  --pvc runai-nlp-magron-scratch:pvc_recep \
  --node-type G10 \
  --interactive

# runai submit \
# # specify your job name
# --name epitron_train_tv0-1 \
# # specify the harbor image to use
# -i ic-registry.epfl.ch/nlp/magron/nlp-magron:tv0-1 \
# # specify amount of CPUs or GPUs needed
# --gpu 1 --cpu 1 \
# # if you need to mount a persistent volume (pvc) (e.g. NFS) specify which one + where to mount it after the colon, to find out the pvc-s avalaible to you, see "kubectl get pvc" in §useful-runai--kubectl-commands
# --pvc <your-pvc>:<your-mount-dest> \ 
# --node-type G10 # specify node type, options shown at https://icitdocs.epfl.ch/display/clusterdocs/Servers+Details 
# # --interactive => makes it interactive
# # --attach => may help attaching VSCode?
# # --command -- run.sh => overrides entrypoints specified by the docker image

runai describe job epitron-train-tv0-$V -p nlp-magron

echo $((V + 1)) > .run_id


