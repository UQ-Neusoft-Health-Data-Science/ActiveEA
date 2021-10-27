

# related directories and filenames
dataset_name=${data_name}_BACH${bachelor_percent}_${model_name}
init_data_dir="${proj_dir}/dataset/${seed}/${task_group}/${dataset_name}/INIT"
variant_data_dir="${proj_dir}/dataset/${seed}/${task_group}/${dataset_name}/${data_variant}"
variant_out_root_dir="${proj_dir}/output/${seed}/${task_group}/${dataset_name}/${data_variant}"
res_dir=${proj_dir}/output/${seed}/results/
metric_dir="${res_dir}/${task_group}/${dataset_name}"
if [ ! -d "${proj_dir}/output/${seed}/${task_group}/" ]; then
    mkdir -p "${proj_dir}/output/${seed}/${task_group}/"
fi
if [ ! -d $metric_dir ]; then
    mkdir -p $metric_dir
fi
metric_fn=${metric_dir}/${data_variant}.csv

if [ -d ${variant_data_dir} ]; then
    rm -r ${variant_data_dir}
fi
cp -r $init_data_dir $variant_data_dir



