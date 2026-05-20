#!/bin/bash

while getopts "d:m:" args
do
    case $args in
        d)
            dataset=$OPTARG
            ;;
        m)
            mode=$OPTARG
            ;;
        *)
            echo "Usage: $0 -d <dataset> -m <mode>"
            exit 1
            ;;
    esac
done

cd ../
for i in {1..5}
do
    echo "Running iteration $i"
    python -m src.main --dataset $dataset --max_iteration 2 --model qwen3-235b-a22b --mode $mode
done