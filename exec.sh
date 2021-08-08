#! /bin/bash

BATCH_SIZE=128
DROPOUT=0.5
BACKBONE=efficientnetb3
EPOCHS=400
TARGET=165
MODEL=FPN
OPTIMIZER=RAdam
LR=0.001
BASENAME=$TARGET-$MODEL-$BACKBONE-$DROPOUT-$BATCH_SIZE-$OPTIMIZER-$LR
CHECKPOINT=checkpoints/$BASENAME.h5
LOG=logs/$(date +%F_%T)-$BASENAME.log

mkdir -p logs checkpoints

python tuto.py --batch_size $BATCH_SIZE \
       --dropout $DROPOUT \
       --backbone $BACKBONE \
       --epochs $EPOCHS \
       --target $TARGET \
       --checkpoint $CHECKPOINT \
       --model $MODEL \
       --optimizer $OPTIMIZER \
       --lr $LR 2>&1 | tee $LOG
