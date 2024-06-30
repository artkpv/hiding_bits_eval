#!/bin/bash

MYENV=".venv"

function setup() {
    python -m venv $MYENV
    source $MYENV/bin/activate
    pip install -r requirements.txt
}

function test() {
	python -m pytest "$@"
}

function train() {
    echo 'Train'
    [[ "$OPENAI_API_KEY" != "" ]] || ( echo "No OpenAI api key" ; exit )
    timestamp=$( date +%Y%m%d_%H-%M-%S )
    ODIR=artifacts/${timestamp}-train
    mkdir $ODIR
    sbatch --export=ALL --output=$ODIR/slurm-%j.out  launch.batch accelerate launch --config_file ./accelerate-no-distributed-no-deepspeed.yaml train.py --output_dir=$ODIR "$@"
    echo 'Waiting job...'
    # Wait till output file appears and then print it
    while [ ! -f $ODIR/slurm-*.out ]; do sleep 1; done
    less $ODIR/slurm-*.out
}

function eval() {
    echo 'Eval'
    [[ "$OPENAI_API_KEY" != "" ]] || ( echo "No OpenAI api key" ; exit )
    timestamp=$( date +%Y%m%d_%H-%M-%S )
    ODIR=artifacts/${timestamp}-eval
    mkdir $ODIR
    sbatch --export=ALL --output=$ODIR/slurm-%j.out  launch.batch accelerate launch --config_file ./accelerate-no-distributed-no-deepspeed.yaml eval.py --output_dir=$ODIR "$@"
    echo 'Waiting job...'
    # Wait till output file appears and then print it
    while [ ! -f $ODIR/slurm-*.out ]; do sleep 1; done
    less $ODIR/slurm-*.out
}

function eval_parallel() {
    models='aichat:openai:gpt-3.5-turbo
aichat:openai:gpt-4-turbo-preview
aichat:openai:gpt-4-1106-preview
aichat:gemini:gemini-pro
aichat:claude:claude-3-sonnet-20240229
' 
    for model in $models ; do
        echo "Evaluating $model"
        > "artifacts/$model.log" python eval.py --model $model --ds_size=500 "$@" & 
        sleep 2 # Wait such that the log folder is different
    done
}

function eval_llama() {
    eval --models "meta-llama/Meta-Llama-3-8B-Instruct" --overseer "aichat:openai:gpt-3.5-turbo-1106"
}


function oaieval() {
    timestamp=$( date +%Y%m%d_%H-%M-%S )
    export EVALS_THREADS=1
    ODIR=artifacts/${timestamp}
    mkdir $ODIR
    sbatch --export=ALL --output=$ODIR/slurm-%j.out launch.batch oaieval --record_path $ODIR/oaieval.jsond "$@"
    echo 'Waiting job...'
    # Wait till output file appears and then print it
    while [ ! -f $ODIR/slurm-*.out ]; do sleep 1; done
    less $ODIR/slurm-*.out
}

if [[ $# == 0 ]] ; then 
    echo "No command provided. Exiting."
else 
    cmd=$1
    shift 1
    $cmd "$@"
fi
