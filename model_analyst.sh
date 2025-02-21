#!/bin/bash

timestamp() {
  date +%Y-%m-%dT%H-%M-%S
}

# names of the Docker containers
DOCKER_CONSTRAINTREE="constraintree"
DOCKER_MEDIUM="lc10000"

# Prepare the Docker environment
docker-compose up -d --build > /dev/null
sleep 120s  # give the containers some time for initialization
docker-compose stop > /dev/null

for config_file in ./SynthLC_Configs/Analyst/*.json; do
  config_name=$(basename -- "$config_file")

  echo "$(timestamp) $config_name"
  docker restart $DOCKER_MEDIUM &> /dev/null
  sleep 30s  # some buffer for the containers to be responsive

  if [[ $config_name == *"_1.json"* ]]; then
    docker restart $DOCKER_CONSTRAINTREE &> /dev/null
#    sleep 30s  # some buffer for the containers to be responsive
    echo "$(timestamp) ${config_name} no_validation"
    docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 902 python execute.py /configs/Analyst/$config_name --no_validation)"

    docker restart $DOCKER_CONSTRAINTREE &> /dev/null
#    sleep 30s  # some buffer for the containers to be responsive
    echo "$(timestamp) ${config_name} heuristic_validation"
    docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 902 python execute.py /configs/Analyst/$config_name --heuristics)"
  fi

  docker restart $DOCKER_CONSTRAINTREE &> /dev/null
#  sleep 30s  # some buffer for the containers to be responsive
  echo "$(timestamp) ${config_name} consider_as_feature"
  docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 902 python execute.py /configs/Analyst/$config_name --heuristics --validation_as_feature)"
done

docker-compose stop
