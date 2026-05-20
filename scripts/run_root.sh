#!/bin/bash

while getopts "d:m:r:t:p:" args
do
    case $args in
        d)
            dataset=$OPTARG
            ;;
        m)
            model=$OPTARG
            ;;
        r)
            root_generation=$OPTARG
            ;;
        t)
            run_iteration=$OPTARG
            ;;
        p)
            parallel_flag=$OPTARG
            ;;
        *)
            echo "Usage: $0 -d <dataset> -m <model> -r <root_generation> -t <run_iteration> -p <parallel_flag>"
            exit 1
            ;;
    esac
done

cd ../
for i in $(seq 1 $run_iteration)
do
    echo "Running iteration $i"
    if [ "$parallel_flag" = "true" ]; then
        python -m src.main --dataset $dataset --max_iteration 2 --model $model --mode average --root_generation $root_generation
    else
        python -m src.main --dataset $dataset --max_iteration 2 --model $model --mode average --root_generation $root_generation --no_parallel
    fi
done