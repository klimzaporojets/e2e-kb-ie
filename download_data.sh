#!/usr/bin/env bash
python -m spacy download en_core_web_sm
# gets and unzips the data from the storage
wget -O data.zip "https://cloud.ilabt.imec.be/index.php/s/42nPg5cPo76Xcnd/download?files=data.zip"
unzip data.zip
rm data.zip
# gets the dwie dataset from git@github.com:klimzaporojets/DWIE.git
mkdir data/datasets/dwie
git clone https://github.com/klimzaporojets/DWIE.git data/datasets/dwie
cd data/datasets/dwie
python src/dwie_download.py --tokenize True
cd ../../../