# Injecting Knowledge Base Information into End-to-End Joint Entity and Relation Extraction and Coreference Resolution 

##(this repository is a work in progress!)

## Introduction

This repository contains the code to reproduce the results from the following paper:
```
@article{verlinden2021injecting,
  title={Injecting Knowledge Base Information into End-to-End Joint Entity and Relation Extraction and Coreference Resolution},
  author={Verlinden, Severine and Zaporojets, Klim and Deleu, Johannes and Demeester, Thomas and Develder, Chris},
  year={2021},
  booktitle={Proceedings of the 2021 Annual Meeting of the Association for Computational Linguistics (ACL 2021): Findings} 
}
```

## Downloading the Data
To download the datasets, embeddings and dictionaries execute the following script: 

```./download_data.sh```

After running the script, the resulting directory structure should be as follows:
```
├── experiments
├── data
│   ├── datasets
│   │   ├── docred
│   │   └── dwie 
│   ├── dictionaries
│   └── embeddings
│       ├── kb
│       └── text
├── src

```

## Creating the Environment
We recommend creating a separate environment to run the code and 
then install the packages in requirements.txt: 
```
conda create -n e2e-kb-ie python=3.9
conda activate e2e-kb-ie
pip install -r requirements.txt
``` 


## Training
The training script is located in ```src/train.py```, it takes two arguments:
 
1- ```--config_file```: the configuration file to run one of the experiments in ```experiments``` directory
 (the names of experiment config files are self explanatory).  
2- ```--path```: the directory where the output model, results and tensorboard logs are going to be 
saved. For each experiment, a separate subdirectory with the name of experiment is automatically 
created. 

Example: 

```python src/train.py --config_file experiments/attention_dwie_kb_both.json --path results``` 

## Evaluation
After the training is finished, the prediction files are saved as ```test.json``` inside the results
directory. In order to obtain the F1 scores execute:  

```python src/evaluation_script.py --pred [path to prediction file] --gold data/datasets/dwie/data/annos_with_content/ --gold-filter test```

Example: 

```python src/evaluation_script.py --pred results/attention_dwie_kb_both/test.json --gold data/datasets/dwie/data/annos_with_content/ --gold-filter test```

The ```test.json``` can be also obtained by explicitly loading and evaluating the model in the results directory: 

```python -u src/run_model.py --path [path to the results directory with serialized model savemodel.pth]```

Example:  

```python -u src/run_model.py --path results/attention_dwie_kb_both/```
