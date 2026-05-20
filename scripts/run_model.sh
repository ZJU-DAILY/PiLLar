#!/bin/bash

while getopts "d:m:t:" args
do
    case $args in
        d)
            dataset=$OPTARG
            ;;
        m)
            model=$OPTARG
            ;;
        t)
            run_iteration=$OPTARG
            ;;
        *)
            echo "Usage: $0 -d <dataset> -m <model> -t <run_iteration>"
            exit 1
            ;;
    esac
done

cd ../
for i in $(seq 1 $run_iteration)
do
    echo "Running iteration $i"
    python -m src.main --dataset $dataset --max_iteration 2 --model $model --mode average
done