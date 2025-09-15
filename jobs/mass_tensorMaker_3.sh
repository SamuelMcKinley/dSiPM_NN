#!/bin/bash
#SBATCH -J "mass_tensorMaker_3.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 1
#SBATCH --mem-per-cpu=8G

# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:$PATH
echo "Environment loaded."

cd ..
home_dir=$PWD
temp_dir=/lustre/scratch/$USER/dSiPM_NN/
cd ${temp_dir}

# Loop as long as the master script is running
while squeue -u "$USER" | grep -q "masterT"; do

  # Look for text file used to communicate between masterTrain.sh and mass_tensorMaker.sh
  # Script will only run when file is found
  if [ ! -f "start_tensorMaker_3.txt" ]; then
    sleep 5
  else
    echo "Found start_tensorMaker_3.txt"

    # Copy over variables: , , , 
    . start_tensorMaker_3.txt

    # Make time slices of xy projected photon tensors from the .root simulation files. Args: <simulation file> <energy> <output tensor folder name> <SPAD Size>
    python3 -u ${home_dir}/tensorMaker.py     mc_Simulation_run0_3_Test_1evt_${particle}_${energy}_${energy}.root     ${energy}     tensor_${group}_3_${particle}_${energy}     ${SPAD_Size}

    # Delete simulation file
    rm -rf mc_Simulation_run0_3_Test_1evt_${particle}_${energy}_${energy}.root

    mv tensor_${group}_3_${particle}_${energy}/npy/* tensfold/tensor_3_${particle}_${energy}_${group}_${SPAD_Size}.npy
    rm -rf tensor_${group}_3_${particle}_${energy}

    echo "DONE"

    # Remove communication text file
    rm -rf start_tensorMaker_3.txt

    # Communicate to masterTrain.sh that the tensor making is finished
    touch tensorMaker_check/3.done
  fi

done

