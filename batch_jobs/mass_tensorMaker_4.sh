#!/bin/bash
#SBATCH -J "mass_tensorMaker_4.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 2
#SBATCH --mem-per-cpu=32G

set -euo pipefail

# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:$PATH
echo "Environment loaded."

cd ..
home_dir=$PWD
temp_dir=/lustre/scratch/$USER/dSiPM_NN/
cd ${temp_dir}

# Loop as long as the master script is running
while squeue -u "$USER" | grep -q "batch_wo"; do

  # Look for text file used to communicate between batch_workflow.py and mass_tensorMaker.sh
  # Script will only run when file is found
  if [ ! -f "start_tensorMaker_4.txt" ]; then
    sleep 1
  else
    echo "Found start_tensorMaker_4.txt"

    # Wait until the communication file is ready and populated
    while true; do
      if [ -s "start_tensorMaker_4.txt" ]; then
        sleep 2
        . start_tensorMaker_4.txt
        echo "✅ Loaded variables: particle=${particle}, energy=${energy}, group=${group}, SPAD_Size=${SPAD_Size}"
        break
      else
        echo "⏳ Waiting for start_tensorMaker_4.txt to be ready..."
        sleep 2
      fi
    done

    # Now that variables are sourced, wait for the ROOT file
    root_file="mc_Simulation_run0_4_Test_1evt_${particle}_${energy}_${energy}.root"
    while [ ! -s "$root_file" ]; do
      echo "⏳ Waiting for $root_file to be written..."
      sleep 3
    done

    # Make time slices of xy projected photon tensors from the .root simulation files. Args: <simulation file> <energy> <output tensor folder name> <SPAD Size>
    python3 -u ${home_dir}/tensorMaker.py     mc_Simulation_run0_4_Test_1evt_${particle}_${energy}_${energy}.root     ${energy}     tensor_${group}_4_${particle}_${energy}     ${SPAD_Size}

    # Delete simulation file
    rm -rf mc_Simulation_run0_4_Test_1evt_${particle}_${energy}_${energy}.root

    mv tensor_${group}_4_${particle}_${energy}/npy/* tensfold/tensor_4_${particle}_${energy}_${group}_${SPAD_Size}.npy
    rm -rf tensor_${group}_4_${particle}_${energy}

    echo "Tensor making complete for particle ${particle}, ${energy} GeV, group ${group}, SPAD_Size ${SPAD_Size}."

    # Remove communication text file
    rm -rf start_tensorMaker_4.txt

    # Communicate to workflow_manager.py that the tensor making is finished
    touch tensorMaking_check/4.done
  fi

done

