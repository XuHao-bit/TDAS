# TDAS
Code of the paper "[CIKM 2023] Task-Difficulty-Aware Meta-Learning with Adaptive Update Strategies for User Cold-Start Recommendation"

## Abstract
User cold-start recommendation is one of the most challenging problems that limit the effectiveness of recommender systems. Meta-learning-based methods are introduced to address this problem by learning initialization parameters for cold-start tasks. Recent studies attempt to enhance the initialization methods. They first represent each task by the cold-start user and interacted items. Then they distinguish tasks based on the task relevance to learn adaptive initialization. However, this manner is based on the assumption that user preferences can be reflected by the interacted items saliently, which is not always true in reality. In addition, we argue that previous approaches suffer from their adaptive framework (e.g., adaptive initialization), which reduces the adaptability in the process of transferring meta-knowledge to personalized RSs. In response to the issues, we propose a task-difficulty-aware meta-learning with adaptive update strategies (TDAS) for user cold-start recommendation. First, we design a task difficulty encoder, which can represent user preference salience, task relevance, and other task characteristics by modeling task difficulty information. Second, we adopt a novel framework with task-adaptive local update strategies by optimizing the initialization parameters with task-adaptive per-step and per-layer hyperparameters. Extensive experiments based on three real-world datasets demonstrate that our TDAS outperforms the state-of-the-art methods.

## Requirements 
- Python 3.7
- pytorch 1.1.0
- numpy 1.21.5
- pandas 1.3.5

## Dataset

1. The raw datasets could be downloaded from: 
- [MovieLens](https://grouplens.org/datasets/movielens/)
- [Bookcrossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
- DBook is provided by MetaHIN [KDD'20], and you can find it from [here](https://github.com/rootlu/MetaHIN)

2. Create a folder named `data_raw` and put the dataset into this folder. 

3. Create a folder named `data_processed`, you can process the raw datasets via
`python prepareDataset.py`

Here we only give the processing code for the MovieLens dataset, please write your own code for processing Bookcrossing and DBook dataset with the similar functions presented in `prepareMovielens.py`

4. The structure of the processed dataset:

```
- data_processed

  - bookcrossing
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
   
  - movielens
    - raw
      sample_1_x1.p
      sample_1_x2.p
      sample_1_y.p
      ...
    item_dict.p
    item_state_ids.p
    ratings_sorted.p
    user_dict.p
    user_state_ids.p
```

5. Our code for data processing are implemented based on the code of [MAMO [KDD'20]](https://github.com/dongmanqing/Code-for-MAMO)

## Model training
The structure of our code: 
```
- prepare_data
  prepareList.py
  prepareMovielens.py
- modules
  BaseRecModel.py: Implementation of the base recommender.
  HyperParamModel.py: Implementation of the task adaptive hyperparameter generator.
  MyOptim.py: Implementation of the Adam optimizer (for global updating the base model).
  TaskEncoder.py: Implementation of the task difficulty encoder.
DataLoading.py
prepareDataset.py
run_experiment.py
TaskDiffModel.py
Trainer.py
utils.py
```

## Start
Change your running config at `run_experiment.py` and `utils.py`, and start a training procedure by:
```
python run_experiment.py --mode tdmeta >> [result path]
```

## Citation 
If you use this code, please consider to cite the following paper:

```
@inproceedings{DBLP:conf/cikm/ZhaoZWJYT23,
  author       = {Xuhao Zhao and
                  Yanmin Zhu and
                  Chunyang Wang and
                  Mengyuan Jing and
                  Jiadi Yu and
                  Feilong Tang},
  title        = {Task-Difficulty-Aware Meta-Learning with Adaptive Update Strategies
                  for User Cold-Start Recommendation},
  booktitle    = {{CIKM}},
  pages        = {3484--3493},
  publisher    = {{ACM}},
  year         = {2023}
}
```
