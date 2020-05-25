#!/bin/bash

mkdir -p snap/pretrained/bert
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz -P snap/pretrained/bert
tar -xvf snap/pretrained/bert/bert-base-uncased.tar.gz
rm snap/pretrained/bert/bert-base-uncased.tar.gz
wget  https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt -P snap/pretrained/bert
mv snap/pretrained/bert/bert-base-vocab.txt snap/pretrained/bert/vocab.txt
