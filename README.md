# DynMo: Elastic Load Balancing for Dynamic LLMs

DynMo is a dynamic load balancer that is built on top of MegatronLM ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) transformer developed by the Applied Deep Learning Research team at NVIDIA. This repo gives an example case for using DynMo: gradual pruning in-training. 

# Contents
   * [Contents](#contents)
   * [Usage](#usage)
   * [Training](#training)
      * [Data Preprocessing](#data-preprocessing)
      * [BERT Pretraining](#bert-pretraining)
   * [Datasets](#datasets)
      * [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)

# Usage

1. Install Dependencies 
2. Data preprocessing
3. Pretraining

We've provided several scripts for pretraining in [`examples`](./examples) directory.

## Installing Dependencies

0. Create an environment (Optional)

```bash
conda create -n myenv python=3.9
conda activate myenv
```

1. Install Sputnik.

```bash
cd sputnik
mkdir build & cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="70;75;80" -DCMAKE_INSTALL_PREFIX=$(pwd)
make -j8 & make install
```

2. Install PyTorch
```bash
pip install torch==2.1.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

3. Install Apex
```bash
cd apex
pip install packaging
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

4. Install anon-repo
This repository contains the binding of Sputnik for PyTorch. Repository link is anonymized for double blind policy.

```bash
git clone https://anonymous.4open.science/r/Torch-Sputnik-E926
cd anon-repo
python setup.py install
```

## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset-impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output-prefix my-bert \
       --vocab bert-vocab.txt \
       --dataset-impl mmap \
       --tokenizer-type BertWordPieceLowerCase \
       --split-sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data-path` specified in later BERT training is the full path and new filename, but without the file extension.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

## LLM Pretraining
bash examples/bert_example.sh 24 diffusion 4 1

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

# Datasets
We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, use the `--split-sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.