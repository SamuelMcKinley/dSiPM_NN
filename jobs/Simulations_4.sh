#!/bin/bash
#SBATCH -J Simulations_4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona

cd ..
home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
temp_dir=/lustre/scratch/$USER/dSiPM_NN

cd ${temp_dir}

# Loop as long as the master script is running
echo "Searching for start_Simulations_4.txt"
while squeue -u "$USER" | grep -q "masterT"; do

  # Look for text file used to communicate between masterTrain.sh and Simulations.sh
  # Script will only run when file is found
  if [ ! -f "start_Simulations_4.txt" ]; then
    sleep 5
  else
    echo "Found start_Simulations_4.txt"

    # Copy over variables: ${particle}, ${energy}
    . start_Simulations_4.txt

    echo "Running simulations inside ${sim_dir}"

    # Run GEANT4 simulation from directory parallel to dSiPM_NN
    singularity exec --cleanenv --bind /lustre:/lustre /lustre/work/yofeng/SimulationEnv/alma9forgeant4_sbox/       bash -c "source /workspace/geant4-v11.2.2-install/bin/geant4.sh &&         ${sim_dir}/exampleB4b -b ${sim_dir}/paramBatch03_single.mac \
        -jobName Simulation -runNumber 0 -runSeq 4 \
        -numberOfEvents 1 -eventsInNtupe 1 \
        -gun_particle $particle -gun_energy_min $energy -gun_energy_max $energy \
        -sipmType 1"

    echo "Simulation complete"

    # Remove communication text file
    rm -rf start_Simulations_4.txt

    # Communicate to masterTrain that the simulation is finished
    touch Simulation_check/4.done
  fi

done
