# ActiveEA

Source code of paper "ActiveEA: Active Learning for Neural Entity Alignment", which has been accepted at EMNLP 2021.

## Steps of reproducing the experiments:

- Step 1: Download and unzip the repo. Suppose it is put under `/path_to_proj/` and we refer the directory under it with relative path. 
- Step 2: If you need to run our strategies on RDGCN, you need to download the open word embedding file 
from [wiki-news-300d-1M.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip) 
and put the unzipped file under `dataset/`.
Otherwise, skip this step (the size of unzipped word embedding file will be 2.26GB).
- Step 3: Install conda environment.
`cd` to your project directory firstly. Then, create the environment using command below.
```shell script
conda env create -f environment.yml
```
Then, activate the environment as `conda activate al4ea`, and install more packages using the following commands

```shell
conda install -c conda-forge graph-tool==2.29
pip install igraph
pip install python-Levenshtein
pip install gensim==4.0.1
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

## Citation

If you re-use our code for your paper, please kindly cite our paper:

```text
@inproceedings{DBLP:conf/emnlp/LiuSZHZ21,
  author    = {Bing Liu and
               Harrisen Scells and
               Guido Zuccon and
               Wen Hua and
               Genghong Zhao},
  editor    = {Marie{-}Francine Moens and
               Xuanjing Huang and
               Lucia Specia and
               Scott Wen{-}tau Yih},
  title     = {ActiveEA: Active Learning for Neural Entity Alignment},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2021, Virtual Event / Punta Cana, Dominican
               Republic, 7-11 November, 2021},
  pages     = {3364--3374},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://doi.org/10.18653/v1/2021.emnlp-main.270},
  doi       = {10.18653/v1/2021.emnlp-main.270},
  timestamp = {Thu, 20 Jan 2022 10:02:11 +0100},
  biburl    = {https://dblp.org/rec/conf/emnlp/LiuSZHZ21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



## Acknowledgement
We implement the neural EA models by customizing source code of [OpenEA](https://github.com/nju-websoft/OpenEA).


