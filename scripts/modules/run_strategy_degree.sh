

params="--data_dir=$variant_data_dir --arg_fn=$al_ea_arg_fn --seed=${seed}"
${py_exe_fn} $proj_dir/al4ea/strategies/degree.py ${params}

