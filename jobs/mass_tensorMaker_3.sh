#!/bin/bash
#SBATCH -J "mass_tensorMaker_3.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 1
#SBATCH --mem-per-cpu=8G

# Replace with your environment setup
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:$PATH
echo "Environment loaded."

cd ..
home_dir=$PWD
temp_dir=/lustre/scratch/$USER/dSiPM_NN/
cd ${temp_dir}

while squeue -u "$USER" | grep -q "masterT"; do

  if [ ! -f "start_tensorMaker_3.txt" ]; then
    sleep 5
  else
    . start_tensorMaker_3.txt
    echo "Initialized for run 3 at particle $particle, energy $energy, group $group, SPAD Size $SPAD_Size"

    python3 -u ${home_dir}/tensorMaker.py     mc_Simulation_run0_3_Test_1evt_${particle}_${energy}_${energy}.root     ${energy}     tensor_${group}_3_${particle}_${energy}     ${SPAD_Size}

    rm -rf mc_Simulation_run0_3_Test_1evt_${particle}_${energy}_${energy}.root

    mv tensor_${group}_3_${particle}_${energy}/npy/* tensfold/tensor_3_${particle}_${energy}_${group}_${SPAD_Size}.npy
    rm -rf tensor_${group}_3_${particle}_${energy}

    echo "DONE"

    rm -rf start_tensorMaker_3.txt

    touch tensorMaker_check/3.done
  fi

done

