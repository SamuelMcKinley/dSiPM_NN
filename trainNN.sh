#!/bin/bash

cd jobs

gen_script() {
    local script_name="NN_training.sh"

    cat << EOF > "NN_training.sh"
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
home_dir=\$PWD
temp_dir=/lustre/scratch/\$USER/dSiPM_NN
train_dir=\${home_dir}/NNTraining

# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."

cd \${temp_dir}


# Loop as long as the master script is running
while squeue -u "\$USER" | grep -q "masterT"; do

  # Look for text files used to communicate between masterTrain.sh and trainNN.sh
  # Script will only run when file is found
  if [ ! -f "start_training.txt" ]; then
    sleep 5
  else
    echo "Fround start_training.txt"

    # Copy over variable: ${SPAD_Size}
    . start_training.txt

    cd \${train_dir}

    # Creates directory. E.G. 70x70 model
    mkdir -p \${SPAD_Size}_model
    cp -r current_model/* \${SPAD_Size}_model
    cd \${SPAD_Size}_model


    # Train entire group to model. Args: <folder with tensors> <energy> --spad <SPAD Size>
    python3 -u train.py \${temp_dir}/tensfold \
    \${energy} --spad \${SPAD_Size}


    cd \${home_dir}

    # Moves the already summed tensor from last iteration, so the other tensors can be summed to it as well
    if [ -d output_\${SPAD_Size}/summed_tensor_\${SPAD_Size} ]; then
      mv output_\${SPAD_Size}/summed_tensor_\${SPAD_Size}/summed_tensor.npy \${temp_dir}/tensfold/
    fi

    # Sums tensors. Useful for plots
    python3 -u combine_tensors.py \${temp_dir}/tensfold summed_tensor_\${SPAD_Size} && echo "combined tensors"

    mkdir -p output_\${SPAD_Size}
    mv \${temp_dir}/tensfold/summed_tensor_\${SPAD_Size} output_\${SPAD_Size}/ && echo "moved tensors"

    rm -rf \${temp_dir}/tensfold/*

    cd \${temp_dir}

    # Remove communication text file
    rm -rf start_training.txt

    # Communicate to masterTrain.sh that the NN training is finished
    touch NNTraining_check/0.done
  fi

done

EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script

