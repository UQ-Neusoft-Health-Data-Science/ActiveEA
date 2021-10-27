#!/bin/bash

cur_dir=$(dirname "${PWD}/${0}")
cd $cur_dir

# select one task setting firstly
setting_fn="$cur_dir/task_settings.sh"
. $setting_fn

echo $cur_dir

# After generating the datasets, then select some strategies to run

tmp_dir=$cur_dir/tmp/
if [ ! -d $tmp_dir ]; then
    mkdir -p $tmp_dir
fi



data_name=D_W_15K_V1
data_size=15K


for model_name in "Alinet" "BootEA"
do
  for bachelor_percent in 0.0 0.3
  do
    tmp_setting_fn=$tmp_dir/${seed}_${task_group}_${data_name}_bach${bachelor_percent}_eamodel${model_name}_generate_dataset.sh
    cat $setting_fn > $tmp_setting_fn
    echo "data_name=${data_name}" >> $tmp_setting_fn
    echo "model_name=${model_name}" >> $tmp_setting_fn
    echo "bachelor_percent=${bachelor_percent}" >> $tmp_setting_fn
    echo "data_size=${data_size}" >> $tmp_setting_fn
    echo "task_group=effect_of_batchsize" >> $tmp_setting_fn

    sh generate_dataset.sh $tmp_setting_fn
  done
done




for model_name in "Alinet" "BootEA"
do
  for bachelor_percent in 0.0 0.3
  do
    for sample_num_per_ite in 100 200 300 400 500 1000
    do
      tmp_setting_fn=$tmp_dir/${seed}_${task_group}_${data_name}_bach${bachelor_percent}_eamodel${model_name}_batchsize${sample_num_per_ite}.sh
      cat $setting_fn > $tmp_setting_fn
      echo "sample_num_per_ite=${sample_num_per_ite}" >> $tmp_setting_fn
      echo "data_name=${data_name}" >> $tmp_setting_fn
      echo "model_name=${model_name}" >> $tmp_setting_fn
      echo "bachelor_percent=${bachelor_percent}" >> $tmp_setting_fn
      echo "data_size=${data_size}" >> $tmp_setting_fn
      echo "task_group=effect_of_batchsize" >> $tmp_setting_fn


      dataset_name=${data_name}_BACH${bachelor_percent}_${model_name}
      init_data_dir="${proj_dir}/dataset/${seed}/${task_group}/${dataset_name}/INIT"

      until [ -d "$init_data_dir" ]; do
        sleep 10
        echo "have not found the initial dataset. keep waiting ..."
      done

      sh run_struct_uncertainty.sh $tmp_setting_fn
      if [ "1" -eq "$(echo "$bachelor_percent > 0.0" | bc -l)" ]; then
        sh run_struct_uncertainty_bachelor_recog.sh $tmp_setting_fn
      fi
    done
  done
done






