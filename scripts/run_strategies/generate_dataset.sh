#!/bin/bash


if [ ! "$1" = "" ]; then
  . "$1"
fi

script_dir=$(dirname "$PWD/${0}")
. $script_dir/../modules/env_settings_${machine}.sh

dataset_name=${data_name}_BACH${bachelor_percent}_${model_name}
init_data_dir="${proj_dir}/dataset/${seed}/${task_group}/${dataset_name}/INIT"

# general settings
src_ea_data_dir="${proj_dir}/dataset/OpenEA_dataset_v1.1/${data_name}"
new_al4ea_data_dir="${proj_dir}/dataset/${seed}/${task_group}/${dataset_name}"
raw_data_dir="${new_al4ea_data_dir}/RAW"


if [ -d $new_al4ea_data_dir ]; then
#  rm -r $new_al4ea_data_dir
 echo "WARNING: the specified directory already exists"
 exit 1
fi
mkdir -p $new_al4ea_data_dir
cp -r $src_ea_data_dir $raw_data_dir
cp -r $raw_data_dir $init_data_dir

${py_exe_fn} $proj_dir/al4ea/data_proc.py --data_dir=$init_data_dir --bachelor_ratio=$bachelor_percent --seed=${seed}

echo COMPLETE
