#! /bin/bash

BATCH_SIZE=16
DROPOUT=0.05
BACKBONE=seresnet152
EPOCHS=200
TARGET=165
MODEL=Linknet
CHECKPOINT=checkpoints/$TARGET-$MODEL-$BACKBONE-$DROPOUT-$BATCH_SIZE.h5
LOG=logs/$(date +%F_%T)-$TARGET-$MODEL-$BACKBONE-$DROPOUT-$BATCH_SIZE.log

mkdir -p logs checkpoints

python tuto.py --batch_size $BATCH_SIZE \
       --dropout $DROPOUT \
       --backbone $BACKBONE \
       --epochs $EPOCHS \
       --target $TARGET \
       --checkpoint $CHECKPOINT \
       --model $MODEL 2>&1 | tee $LOG
