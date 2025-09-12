#!/bin/bash
#SBATCH -J NN_training.sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p matador
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1

cd ..
home_dir=$PWD
temp_dir=/lustre/scratch/$USER/dSiPM_NN
train_dir=${home_dir}/NNTraining

# Replace with your environment setup
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:$PATH
echo "Environment loaded."

cd ${temp_dir}


echo "Waiting for text file"
while squeue -u "$USER" | grep -q "masterT"; do

# Inititalizes start.txt to begin and copies variables from masterTrain.sh
  if [ ! -f "start_training.txt" ]; then
    sleep 5
  else
    . start_training.txt
    echo "Initialized for  particle $particle, energy $energy, group $group, SPAD Size $SPAD_Size"
    cd ${train_dir}

    # Creates directory. E.G. 70x70 model
    mkdir -p ${SPAD_Size}_model
    cp -r current_model/* ${SPAD_Size}_model
    cd ${SPAD_Size}_model


    # Train entire group to model
    python3 -u train.py ${temp_dir}/tensfold     ${energy} --spad ${SPAD_Size}


    cd ${home_dir}

    # Moves the already summed tensor from last iteration, so the other tensors can be summed to it as well
    if [ -d output_${SPAD_Size}/summed_tensor_${SPAD_Size} ]; then
      mv output_${SPAD_Size}/summed_tensor_${SPAD_Size}/summed_tensor.npy ${temp_dir}/tensfold/
    fi

    # Sums tensors. Useful for plots
    python3 -u combine_tensors.py ${temp_dir}/tensfold summed_tensor_${SPAD_Size} && echo "combined tensors"

    mkdir -p output_${SPAD_Size}
    mv ${temp_dir}/tensfold/summed_tensor_${SPAD_Size} output_${SPAD_Size}/ && echo "moved tensors"

    rm -rf ${temp_dir}/tensfold/*

    cd ${temp_dir}
    rm -rf start_training.txt
    touch NNTraining_check/0.done
  fi

done

