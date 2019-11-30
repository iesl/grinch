#!/usr/bin/env bash

set -exu

script_to_run=$1
dataset=$2
alg_name=$3
run_id=${4:-1}
threads=${5:-4}
mem=${6:-20000}

TIME=`(date +%Y-%m-%d-%H-%M-%S-%3N)`

log_dir=logs/$dataset/$alg_name/$TIME

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

mkdir -p $log_dir/

partition=cpu

sbatch -J grinch-$dataset-$alg_name-$TIME \
            -e $log_dir/cluster.err \
            -o $log_dir/cluster.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem=$mem \
            --time=0-05:00 \
            $script_to_run "exp_out" $run_id
