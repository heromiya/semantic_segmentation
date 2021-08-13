#! /bin/bash

mkdir -p 165+166/ann 165+166/ann_sample 165+166/img 165+166/img_sample

function m_copy() {
    NUM=$1
    FILE=$2
    DIST=$(echo $FILE | sed "s/[0-9]\{3\}\/\(.*\)\([0-9]\{7\}.tif\)/165+166\/\1${NUM}-\2/g")
    cp $FILE $DIST
}
export -f m_copy

for NUM in 165 166; do
    FILELIST=$(mktemp)
    find $NUM -type f -regex ".*\.tif" > $FILELIST
    parallel m_copy $NUM :::: $FILELIST
done
