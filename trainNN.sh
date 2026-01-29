#!/bin/bash

cd batch_jobs

gen_script() {
    local script_name="NN_training.sh"

    cat << EOF > "NN_training.sh"
#!/bin/bash
#SBATCH -J "trainNN.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p matador
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1

set -euo pipefail

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
while squeue -u "\$USER" | grep -q "batch_wo"; do

  # Look for text files used to communicate between batch_workflow.py and trainNN.sh
  # Script will only run when file is found
  if [ ! -f "start_training.txt" ]; then
    sleep 3
  else
    echo "Found start_training.txt"

    # Copy over variable: ${SPAD_Size}
    . start_training.txt

    cd \${train_dir}

    # Creates directory. E.G. 70x70 model
    mkdir -p \${SPAD_Size}_model
    cp -r current_model/* \${SPAD_Size}_model
    cd \${SPAD_Size}_model

    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export TORCH_NUM_THREADS=1

    # Train entire group to model. Args: <folder with tensors> --spad <SPAD Size>
    python3 -u train.py \${temp_dir}/tensfold --epochs 30 --spad \${SPAD_Size} --bs 1 --workers 0 --early-stop 6 --lr 3e-4

    # Combine tensors
    cd \${home_dir}
    python3 -u combine_tensors.py \${temp_dir}/tensfold \${SPAD_Size}

    rm -rf \${temp_dir}/tensfold/*.npz

    cd \${temp_dir}

    # Remove communication text file
    rm -rf start_training.txt

    # Communicate to workflow_manager.py that the NN training is finished
    touch NNTraining_check/0.done
  fi

done

EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script