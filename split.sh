#! /bin/bash

TARGET=$1
SPLIT=0.27
N_TOTAL=$(find $TARGET/img/ -type f | wc -l)

mkdir -p $TARGET/ann_sample $TARGET/img_sample
mv $TARGET/ann_sample/* $TARGET/ann
mv $TARGET/img_sample/* $TARGET/img

N_VALIDATION=$(echo "$N_TOTAL * $SPLIT" | bc | sed 's/\.[0-9]*$//')

TEMP=$(mktemp)
find $TARGET -type f | shuf | head -n $N_VALIDATION | sed 's/.*\([0-9]\{7\}\).*/\1/g' > $TEMP
parallel mv $TARGET/ann/{}.tif $TARGET/ann_sample/ :::: $TEMP
parallel mv $TARGET/img/{}.tif $TARGET/img_sample/ :::: $TEMP
