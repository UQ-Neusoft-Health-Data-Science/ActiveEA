#!/bin/bash


## ==>> default settings
# options: D_W_15K_V1, EN_DE_15K_V1, EN_FR_100K_V1
data_name=D_W_15K_V1
# options: 15K, 100K, depend on `data_name`
data_size=15K
bachelor_percent=0.3
# Alinet, BootEA, RDGCN
model_name=Alinet
seed=1011


if [ "$data_size" = "15K" ]; then
  sample_num_per_ite=100
elif [ "$data_size" = "100K" ]; then
  sample_num_per_ite=1000
fi
alpha=0.1
dropout_ratio=0.1


## ==>> running env settings
export CUDA_VISIBLE_DEVICES="1"


# task, model, and data size, overall_perf, effect_of_alpha, effect_of_batchsize, effect_of_bachpercent
task_group=overall_perf


# options: server, wiener
machine=server

proj_dir=""
if [ "$proj_dir" = "" ]; then
    echo "ERROR: please set proj_dir in 'task_settings.sh' firstly"
    exit 1
fi

