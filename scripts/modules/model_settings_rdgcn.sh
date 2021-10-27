

# options: Alinet, BootEA, RDGCN
model_name=RDGCN
al_ea_arg_fn="${proj_dir}/openea_run/args/al_rdgcn_args_${data_size}.json"
al_ea_arg_dropout_fn="${proj_dir}/openea_run/args/al_rdgcn_args_${data_size}_dropout.json"
ea_arg_fn="${proj_dir}/openea_run/args/rdgcn_args_${data_size}.json"

if [ ! -f "${proj_dir}/dataset/${seed}/${task_group}/wiki-news-300d-1M.vec" ]; then
  ln -s "${proj_dir}/dataset/wiki-news-300d-1M.vec"  "${proj_dir}/dataset/${seed}/${task_group}/wiki-news-300d-1M.vec"
fi

