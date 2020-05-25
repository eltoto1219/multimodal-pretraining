# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=snap/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/test.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python3 src/tasks/gqa.py \
    --test testdev \
    --llayers 9 --xlayers 1 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --load snap/gqa/gqa_lxr955/BEST \
    --batchSize 2 --optim bert --lr 1e-5 --epochs 1 \
    --tqdm --output $output ${@:3}
