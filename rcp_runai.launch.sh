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
  --pvc runai-nlp-magron-scratch:/home/magron/scratch \
  --node-type G10 \
  --interactive


runai describe job epitron-train-tv0-$V -p nlp-magron

echo $((V + 1)) > .run_id


