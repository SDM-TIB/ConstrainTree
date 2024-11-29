#!/bin/bash

timestamp() {
  date +%Y-%m-%dT%H-%M-%S
}

# number of runs
RUNS=5

# names of the Docker containers
DOCKER_CONSTRAINTREE="constraintree"
DOCKER_SMALL="lc1000"
DOCKER_MEDIUM="lc10000"
DOCKER_LARGE="lc100000"

# Prepare the Docker environment
docker-compose up -d --build > /dev/null
sleep 120s  # give the containers some time for initialization
docker-compose stop > /dev/null

for ((i=1;i<=RUNS;i++)); do
  for config_file in $(echo "./SynthLC_Configs/SPARQLConstraint/*.json"); do
    config_name=$(basename -- "$config_file")
    echo $(timestamp) $config_name $i
    docker restart $DOCKER_CONSTRAINTREE &> /dev/null

    # check which KG to restart
    if [[ $config_name == *"_1000_"* ]]; then
      container=$DOCKER_SMALL
    elif [[ $config_name == *"_10000_"* ]]; then
      container=$DOCKER_MEDIUM
    else
      container=$DOCKER_LARGE
    fi

    if [[ $config_name == *"_1.json"* ]]; then
      docker restart $container &> /dev/null
      sleep 30s  # some buffer for the containers to be responsive
      echo $(timestamp) $config_name $i "no_validation"
      docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 3602 python execute.py /configs/SPARQLConstraint/$config_name --no_validation)"

      docker restart $container &> /dev/null
      sleep 30s  # some buffer for the containers to be responsive
      echo $(timestamp) $config_name $i "naive_validation"
      docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 3602 python execute.py /configs/SPARQLConstraint/$config_name)"
    fi

    docker restart $container &> /dev/null
    sleep 30s  # some buffer for the containers to be responsive
    echo $(timestamp) $config_name $i "heuristic_validation"
    docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 3602 python execute.py /configs/SPARQLConstraint/$config_name --heuristics)"
  done
done

for ((i=1;i<=RUNS;i++)); do
  for config_file in $(echo "./SynthLC_Configs/TargetQuery/*.json"); do
    config_name=$(basename -- "$config_file")
    echo $(timestamp) $config_name $i
    docker restart $DOCKER_CONSTRAINTREE &> /dev/null

    # check which KG to restart
    if [[ $config_name == *"_1000_"* ]]; then
      container=$DOCKER_SMALL
    elif [[ $config_name == *"_10000_"* ]]; then
      container=$DOCKER_MEDIUM
    else
      container=$DOCKER_LARGE
    fi
    
    if [[ $config_name == *"_1.json"* ]]; then
      docker restart $container &> /dev/null
      sleep 30s  # some buffer for the containers to be responsive
      echo $(timestamp) $config_name $i "naive_validation"
      docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 3602 python execute.py /configs/TargetQuery/$config_name)"
    fi

    docker restart $container &> /dev/null
    sleep 30s  # some buffer for the containers to be responsive
    echo $(timestamp) $config_name $i "heuristic_validation"
    docker exec -it $DOCKER_CONSTRAINTREE bash -c "(timeout -s 15 3602 python execute.py /configs/TargetQuery/$config_name --heuristics)"
  done
done

docker-compose stop
