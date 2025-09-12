#!/bin/bash

particle=$1
energy=$2
group=$3
SPAD_Size=$4


# Combine tensors for graphing
cd temp_files/tensfold
echo "Combining and deleting tensors"
python3 -u ../../combine_tensors.py \
. \
tens_${particle}_${SPAD_Size}


# Delete tensor files
for i in $(seq 0 99); do
    rm -rf tens_${i}.npy
done

