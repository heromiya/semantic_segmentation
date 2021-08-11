#! /bin/bash

BATCH_SIZE=$1
DROPOUT=$2
TARGET=$3
OPTIMIZER=$4
PRETRAINED=$5

BACKBONE=resnext101
EPOCHS=400
MODEL=FPN
LR=0.001
LOSS=jd
ACTIVATION=sigmoid

BASENAME=$(date +%F_%T)-$TARGET-$MODEL-$BACKBONE-$DROPOUT-$BATCH_SIZE-$OPTIMIZER-$LR-$LOSS-$ACTIVATION
CHECKPOINT=checkpoints/$BASENAME.h5
LOG=logs/$BASENAME.log

mkdir -p logs checkpoints


python tuto.py --batch_size $BATCH_SIZE \
       --dropout $DROPOUT \
       --backbone $BACKBONE \
       --epochs $EPOCHS \
       --target $TARGET \
       --checkpoint $CHECKPOINT \
       --model $MODEL \
       --optimizer $OPTIMIZER \
       --lr $LR \
       --loss $LOSS \
       --activation $ACTIVATION \
       --pretrained $PRETRAINED 2>&1 | tee $LOG
