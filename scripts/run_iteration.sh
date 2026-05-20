#!/bin/bash

while getopts "d:i:t:" args
do
    case $args in
        d)
            dataset=$OPTARG
            ;;
        i)
            iteration=$OPTARG
            ;;
        t)
            run_iteration=$OPTARG
            ;;
        *)
            echo "Usage: $0 -d <dataset> -i <max_iteration> -t <run_iteration>"
            exit 1
            ;;
    esac
done

cd ../
for i in $(seq 1 $run_iteration)
do
    echo "Running iteration $i"
    python -m src.main --dataset $dataset --max_iteration $iteration --model qwen3-235b-a22b --mode average
done