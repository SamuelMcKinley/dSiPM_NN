#!/bin/bash
#SBATCH -J Simulations_3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH --mem=8G

cd ..
home_dir=$PWD
sim_dir=${home_dir}/../DREAMSim/sim/build
temp_dir=/lustre/scratch/$USER/dSiPM_NN

cd ${temp_dir}

# Loop as long as the master script is running
echo "Searching for start_Simulations_3.txt"
while squeue -u "$USER" | grep -q "batch_wo"; do

  # Look for text file used to communicate between batch_workflow.py and Simulations.sh
  # Script will only run when file is found
  if [ ! -f "start_Simulations_3.txt" ]; then
    sleep 1
  else
    echo "Found start_Simulations_3.txt"

    # Copy over variables: ${particle}, ${energy}
    . start_Simulations_3.txt

    echo "Running simulations inside ${sim_dir}"

    # Generate seeds
    seed1=$(( (RANDOM << 15) + RANDOM ))
    seed2=$(( (RANDOM << 15) + RANDOM ))

    # Build temporary macro for seeds
    echo "/random/setSeeds $seed1 $seed2" > random_3.mac
    cat ${sim_dir}/paramBatch03_single.mac >> random_3.mac

    # Run GEANT4 simulation from directory parallel to dSiPM_NN
    singularity exec --cleanenv --bind /lustre:/lustre /lustre/work/yofeng/SimulationEnv/alma9forgeant4_sbox/       bash -c "source /workspace/geant4-v11.2.2-install/bin/geant4.sh &&         ${sim_dir}/exampleB4b -b random_3.mac \
        -jobName Simulation -runNumber 0 -runSeq 3 \
        -numberOfEvents 1 -eventsInNtupe 1 \
        -gun_particle $particle -gun_energy_min $energy -gun_energy_max $energy \
        -sipmType 1"

    echo "Simulation complete (seeds: $seed1, $seed2)"


    # Remove communication text file
    rm -rf start_Simulations_3.txt

    # Communicate to workflow_manager.py that the simulation is finished
    touch Simulation_check/3.done
  fi

done
