#!/bin/bash

script_dir=$(dirname "${PWD}/${0}")
cd $script_dir
proj_dir=$script_dir/../../

# select one task setting firstly
setting_fn="$script_dir/task_settings.sh"
. $setting_fn

# then generate initial dataset if we need
sh generate_dataset.sh $setting_fn


## select some strategies to run
sh run_rand.sh $setting_fn
sh run_degree.sh $setting_fn
sh run_pagerank.sh $setting_fn
sh run_between.sh $setting_fn
sh run_uncertainty.sh $setting_fn
sh run_struct_uncertainty.sh $setting_fn

if [ ! $model_name = "BootEA" ]; then
  sh run_deep_uncertainty.sh $setting_fn
  sh run_deep_struct_uncertainty.sh $setting_fn
fi


if [ 1 -eq $(echo "$bachelor_percent > 0.0" | bc -l) ]; then
  sh run_struct_uncertainty_bachelor_recog.sh $setting_fn
  if [ ! $model_name = "BootEA" ]; then
    sh run_deep_struct_uncertainty_bachelor_recog.sh $setting_fn
  fi
fi





