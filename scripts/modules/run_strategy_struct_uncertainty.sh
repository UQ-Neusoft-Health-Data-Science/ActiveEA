

params="--data_dir=${variant_data_dir} --arg_fn=${al_ea_arg_fn} --uncertainty_measure=margin --pr_alpha=${alpha} --edge_mode=basic --sample_num_per_ite=${sample_num_per_ite} --seed=${seed}"
${py_exe_fn} $proj_dir/al4ea/strategies/struct_uncertainty.py ${params}

