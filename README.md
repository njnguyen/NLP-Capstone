# Embedding Structure Matters

This repository contains the software for [Embedding structure matters: Comparing methods to adapt multilingual vocabularies to new languages](https://arxiv.org/abs/2309.04679), to appear in the [3rd Workshop on Multilingual Representation Learning](https://sigtyp.github.io/ws2023-mrl.html) at EMNLP 2023. The general structure of the repository is as follows:

- `src` contains the Python source code for conducting Language-Adaptive Pre-Training (LAPT), fine-tuning and evaluating models on downstream tasks (POS and NER), and re-initializing model embeddings as described in the paper
- `configs` contains YAML configuration files for each experiment in the study (LAPT and fine-tuning)
- `output` contains output logs from evaluation experiments, including scores
- `scripts` contains top-level shell scripts for common routines such as running LAPT, fine-tuning, sampling multilingual sets, and training vocabularies
- `tools` contains auxiliary software fulfilling miscillaneous functions such as pre-processing data, training sentencepiece models, and visualizing embeddings with PCA

## Environment Setup
To manage your Python environment, we recommend you [install anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html). Conda should then be used to create an environment with **Python 3.10**, using this command `conda create --name txlm python=3.10`.

After activating your new environment with `conda activate txlm` or `source activate txlm`, confirm that the result of the command `which pip` returns the path to the `pip` executable within your environment folder, e.g. `~/miniconda3/envs/txlm/bin`.

Next, use conda/pip to install the version of PyTorch that is compatible with your system / CUDA version. Original experiments were conducted with PyTorch version 2.2.2 for CUDA 12.2. The command to install this version is `conda install pytorch torchvision torchaudio pytorch-cuda=12.2 -c pytorch -c nvidia`

Finally, in the main folder of the repository, run the command `pip install -r requirements.txt` to install the required packages.

# Monolingual experiments
## Get Dataset
To download the training dataset for each language, the usage is: 

`./download_oscar.sh path_to_conda_folder environment_name language_code huggingface_access_token output_name`

For example, if you want to download the OSCAR dataset for Telugu, your miniconda folder is at the path `~/username/miniconda3`, the environment is named `txlm`, your Hugging Face access tokens is `abc`, and you want to name your the output file to be `te_oscar.txt`, the command is:

`./download_oscar.sh ~/username/miniconda3 txlm te abc te_oscar.txt`.

## Preprocess Dataset
### Clean Dataset
To clean your dataset (combine the datasets when doing multilingual experiments), and you wish to use Cuda device 0, the usage is:

`./tools/combine_and_clean.sh output_name 0 input_file_1 input_file2 ...`

For example, if you only want to parse the dataset for Telugu, you want the output file name to be `te_output`, and you wish to use Cuda device 0, the command is:

`./tools/combine_and_clean.sh te_output 0 te_oscar.txt`

### Split
Due to the limited computational resources, we split the dataset into 10 shards. The command is:

`./scripts/make_shards.sh te_output_combined_cleaned.txt shards 10`

### Train, Dev, Test set
To get the training, development, and test set, the usage is:

`python train_test_split.py input_dataset_file train_set_portion dev_set_portion`

For example, if your input dataset file is named as `input.txt`, and you want to split this dataset into 0.7, 0.2, 0.1 for training, development, and test set respectively, the command is:

`python train_test_split.py input.txt 0.7 0.2`

### Dataset repository
When you check each configs file, there are some paths to the training dataset and development dataset. You can keep or change those paths. If you choose to keep those paths, you need to create those directories manually and put the dataset file into those directories. 

## Tokenizer and Embeddings
### Vocabulary
To get the vocabulary and tokenizer, assuming you want to get the vocabulary and tokenizer for Telugu, the usage is:

`./scripts/train_vocab.sh ~/username/miniconda3 txlm configs/te_vocab.yml `

### Embeddings
To get the embeddings for each re-initialized embedding technique and focus, assuming you want to find the embeddings for Telugu, ran the command:

`./scripts/initialize_embeddings.sh ~/username/miniconda3 txlm te_spm_32k.model te`

You can take a look at `initialized_embeddings.sh` for more details. If you don't want to initialize the embeddings of some re-initialized embedding techniques, you can modify this file by commenting out the related code lines.

#### Re-initializing Embeddings
The Python script for re-initializing embeddings can be used as follows:

```
python src/reinitialize_embeddings.py \
    --old_model_path path_to_base_model \
    --new_vocab_file path_to_new_vocab \
    --embedding_output_path path_to_new_embedding_matrix \
    [--old_tokenizer_path path_to_base_tokenizer] \
    [--reinit_by_script] \
    [--reinit_by_identity] \
    [--reinit_by_position] \
    [--focus_reinit] \
    [--focus_train_path path_to_focus_training_corpus]
```

`old_model_path` is the name of the base model for which you are re-initializing the vocabulary. For instance, XLM-R: `xlm-roberta-base`. `new_vocab_file` is the sentencepiece model for the new (specialized) vocabulary/tokenizer. `embedding_output_path` is the path at which to save the resulting embedding block (as a PyTorch data file). `old_tokenizer_path` is the optional path to the base (non-specialized) vocabulary/tokenizer, if it is at a different path than the base model (for `xlm-roberta-base`, the model and tokenizer path are the same). The `reinit_by_<method>` arguments are boolean flags for which technique to use for re-initializing embeddings; see our paper for details of these methods. `reinit_by_position` requires that `reinit_by_script` is also true.

Finally `focus_reinit` is the boolean flag to re-initialize embeddings by the [FOCUS method](https://arxiv.org/abs/2305.14481) (see paper for details). This method overrides all other re-initialization methods. It also requires a path to the training corpus for the FOCUS method, via the `focus_train_path` argument. The source code for FOCUS is **not included in this repository**. To use this method, place the [FOCUS Python source code](https://github.com/konstantinjdobler/focus) in a folder called `src/focus`. The FOCUS code requires additional dependencies: `entmax, fasttext, requests, loguru`.


## Running Experiments

### LAPT
Language-Adaptive Pre-training can be run with `scripts/train_model.sh`. The usage is:

`./scripts/train_model.sh path_to_conda_folder environment_name cuda_devices config_name`

For instance, if your miniconda folder is at the path `~/username/miniconda3`, the environment is named `txlm`, you wish to use Cuda devices 0 and 1, and the experiment configuration is `erzya_cpt-full.yml`, the command is:

`./scripts/train_model.sh ~/username txlm 0,1 configs/erzya_cpt-full.yml`

**NOTE**: We use the term Language-Adaptive Pre-Training (LAPT) in the associated publication and readme. However, in configuration file names this is referred to as CPT (Continued Pre-Training).

### Fine-tuning and Evaluation
Similarly, the usage for fine-tuning and evaluating a model on a downstream task is:

`./scripts/eval_finetune.sh path_to_conda_folder environment_name cuda_devices config_name`

The choice of downstream task and hyperparameters is specified in the configuration file. Please see the `configs` folder for examples.
