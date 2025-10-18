#!/bin/bash
#SBATCH -J "mass_tensorMaker_2.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 2
#SBATCH --mem-per-cpu=32G

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
  if [ ! -f "start_tensorMaker_2.txt" ]; then
    sleep 1
  else
    echo "Found start_tensorMaker_2.txt"

    # Copy over variables: , , , 
    . start_tensorMaker_2.txt

    # Make time slices of xy projected photon tensors from the .root simulation files. Args: <simulation file> <energy> <output tensor folder name> <SPAD Size>
    python3 -u ${home_dir}/tensorMaker.py     mc_Simulation_run0_2_Test_1evt_${particle}_${energy}_${energy}.root     ${energy}     tensor_${group}_2_${particle}_${energy}     ${SPAD_Size}

    # Delete simulation file
    rm -rf mc_Simulation_run0_2_Test_1evt_${particle}_${energy}_${energy}.root

    mv tensor_${group}_2_${particle}_${energy}/npy/* tensfold/tensor_2_${particle}_${energy}_${group}_${SPAD_Size}.npy
    rm -rf tensor_${group}_2_${particle}_${energy}

    echo "Tensor making complete for particle ${particle}, ${energy} GeV, group ${group}, SPAD_Size ${SPAD_Size}."

    # Remove communication text file
    rm -rf start_tensorMaker_2.txt

    # Communicate to workflow_manager.py that the tensor making is finished
    touch tensorMaking_check/2.done
  fi

done

