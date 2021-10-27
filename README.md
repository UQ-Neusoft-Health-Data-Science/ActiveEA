# ActiveEA

Source code of paper "ActiveEA: Active Learning for Neural Entity Alignment", which has been accepted at EMNLP 2021.

## Steps of reproducing the experiments:

- Step 1: Download and unzip the repo. Suppose it is put under `/path_to_proj/` and we refer the directory under it with relative path. 
- Step 2: If you need to run our strategies on RDGCN, you need to download the open word embedding file 
from [wiki-news-300d-1M.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) 
and put the unzipped file under `dataset/`.
Otherwise, skip this step (the size of unzipped word embedding file will be 2.26GB).
- Step 3: Install conda environment
```shell script
cd /path_to_proj/al4ea/
conda env create -f environment.yml
```

- Step 4: Configure settings.
The scripts to run are under `scripts/run_strategies/`
The default settings are set in `task_settings.sh`. Before you run any script, set `proj_dir` in the setting file firstly. 


- Step 5: Run scripts:
    * For trials: customizing script `task_runner_trial.sh`.
    * Run experiments about the "overall performance on 15K data": `task_runner_overall_perf.sh`.
    * Run experiments about the "overall performance on 15K data": `task_runner_overall_perf_100k.sh`.
    * Run experiments about the "effect of bachelors": `task_runner_effect_of_bachelor_percent.sh`.
    * Run experiments about the "effectiveness of bachelor recognizer": intermediate results have been saved with the generated dataset of AL process. 
    * Run experiments about the "sensitivity of parameters": `task_runner_effect_of_alpha.sh` and `task_runner_effect_of_batchsize.sh`.

The generated datasets by different AL strategies will be saved to `dataset/` with naming pattern like `dataset/${seed}/${task_group}/${dataset_name}/${strategy_name}`. 
The evaluation results on test set will be saved to `output/results/`. 


## Acknowledgement
We implement the neural EA models by customizing source code of [OpenEA](https://github.com/nju-websoft/OpenEA).


