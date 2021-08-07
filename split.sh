#! /bin/bash

TARGET=166
SPLIT=0.2
N_TOTAL=$(find $TARGET/img/ -type f | wc -l)

mkdir -p $TARGET/ann_sample $TARGET/img_sample

N_VALIDATION=$(echo "$N_TOTAL * $SPLIT" | bc | sed 's/\.[0-9]*$//')

TEMP=$(mktemp)
find $TARGET -type f | shuf | head -n $N_VALIDATION | sed 's/.*\([0-9]\{7\}\).*/\1/g' > $TEMP
parallel mv $TARGET/ann/{}.tif $TARGET/ann_sample/ :::: $TEMP
parallel mv $TARGET/img/{}.tif $TARGET/img_sample/ :::: $TEMP

