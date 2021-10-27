

params="--data_dir=$variant_data_dir --arg_fn=$al_ea_arg_dropout_fn --seed=${seed} --uncertainty_measure=margin --sample_num_per_ite=${sample_num_per_ite}"
${py_exe_fn} $proj_dir/al4ea/strategies/uncertainty.py ${params}

