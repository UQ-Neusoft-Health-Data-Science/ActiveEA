#!/bin/bash

if [ ! "$1" = "" ]; then
  . "$1"
fi

script_dir=$(dirname "$PWD/${0}")
. $script_dir/../modules/env_settings_${machine}.sh


lower_model_name="$(echo $model_name | tr '[A-Z]' '[a-z]')"
. ${proj_dir}/scripts/modules/model_settings_${lower_model_name}.sh



# strategy-specific settings
data_variant=RAND



. $proj_dir/scripts/modules/fn_settings.sh

. $proj_dir/scripts/modules/run_strategy_rand.sh

. $proj_dir/scripts/modules/evaluate_strategy.sh





