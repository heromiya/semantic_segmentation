#! /bin/bash

BATCH_SIZE=512
DROPOUT=0.5
BACKBONE=resnext101
EPOCHS=400
TARGET=165
MODEL=FPN
OPTIMIZER=Adam
LR=0.001
LOSS=jd
ACTIVATION=hard_sigmoid

BASENAME=$TARGET-$MODEL-$BACKBONE-$DROPOUT-$BATCH_SIZE-$OPTIMIZER-$LR-$LOSS-$ACTIVATION
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
       --lr $LR \
       --loss $LOSS \
       --activation $ACTIVATION 2>&1 | tee $LOG
