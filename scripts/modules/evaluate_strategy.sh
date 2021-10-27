

anno_percentages="0.00 0.05 0.10 0.15 0.20 0.30 0.40 0.50"
# training ea model on the annotated data
for ratio in ${anno_percentages}
do
  ${py_exe_fn} ${proj_dir}/openea_run/main_for_al.py --arg_fn=$ea_arg_fn --data_dir=${variant_data_dir} \
 --division=${ratio} --out_dir=${variant_out_root_dir}/res_${ratio}/
done


# gather results
${py_exe_fn} ${proj_dir}/al4ea/gather_results.py --out_dir=${variant_out_root_dir} --metric_fn=${metric_fn}




