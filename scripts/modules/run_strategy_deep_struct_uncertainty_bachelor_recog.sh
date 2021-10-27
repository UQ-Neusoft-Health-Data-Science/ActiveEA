
bach_recog_arg_fn="${proj_dir}/openea_run/args/entity_classifier1_args_15K.json"
params="--data_dir=$variant_data_dir --arg_fn=$al_ea_arg_dropout_fn --uncertainty_measure=margin --pr_alpha=${alpha} \
--bach_recog_arg_fn=${bach_recog_arg_fn} --edge_mode=basic --sample_num_per_ite=${sample_num_per_ite} --seed=${seed}"
${py_exe_fn} $proj_dir/al4ea/strategies/struct_uncertainty_bachelor_recog_simi_thr_cv.py ${params}


