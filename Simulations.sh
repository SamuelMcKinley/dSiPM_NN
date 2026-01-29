#!/bin/bash

cd batch_jobs
Group_Size=$1

# Make the amount of scripts as there are groups
for ((i = 0; i < Group_Size; i++)); do

gen_script() {
    local script_name="Simulations_${i}.sh"

    cat << EOF > "$script_name"
#!/bin/bash
#SBATCH -J Simulations_${i}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH --mem=8G

cd ..
home_dir=\$PWD
sim_dir=\${home_dir}/../DREAMSim/sim/build
temp_dir=/lustre/scratch/\$USER/dSiPM_NN

cd \${temp_dir}

export TMPDIR=/lustre/scratch/\$USER/tmp_\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR"
trap 'rm -rf "\$TMPDIR"' EXIT

# Loop as long as the master script is running
echo "Searching for start_Simulations_${i}.txt"
while squeue -u "\$USER" | grep -q "batch_wo"; do

  # Look for text file used to communicate between batch_workflow.py and Simulations.sh
  # Script will only run when file is found
  if [ ! -f "start_Simulations_${i}.txt" ]; then
    sleep 1
  else
    echo "Found start_Simulations_${i}.txt"

    # Copy over variables: \${particle}, \${energy}
    dos2unix -q "start_Simulations_${i}.txt" 2>/dev/null || true
    . "./start_Simulations_${i}.txt"
    export particle energy

    echo "Running simulations inside \${sim_dir}"

    # Generate seeds
    seed1=\$(( (RANDOM << 15) + RANDOM ))
    seed2=\$(( (RANDOM << 15) + RANDOM ))

    # Build temporary macro for seeds
    echo "/random/setSeeds \$seed1 \$seed2" > random_${i}.mac
    cat \${sim_dir}/paramBatch03_single.mac >> random_${i}.mac

    echo "DEBUG: particle='\$particle' energy='\$energy' sim_dir='\$sim_dir'"
    ls -l "\${sim_dir}/exampleB4b" || echo "MISSING: \${sim_dir}/exampleB4b"

    # Run GEANT4 simulation from directory parallel to dSiPM_NN
    singularity exec --cleanenv \
      --bind /lustre:/lustre \
      --bind "\$TMPDIR":/tmp \
      /lustre/work/yofeng/SimulationEnv/alma9forgeant4_sbox/ \
      bash -c "source /workspace/geant4-v11.2.2-install/bin/geant4.sh && \
        \${sim_dir}/exampleB4b -b random_${i}.mac \\
        -jobName Simulation -runNumber 0 -runSeq ${i} \\
        -numberOfEvents 1 -eventsInNtupe 1 \\
        -gun_particle \"\$particle\" -gun_energy_min \"\$energy\" -gun_energy_max \"\$energy\" \\
        -sipmType 1"
    sim_rc=\$?

    if [ \$sim_rc -eq 0 ]; then
      echo "Simulation complete (seeds: \$seed1, \$seed2)"

      # Remove communication text file only on success
      rm -rf start_Simulations_${i}.txt

      # Communicate success to workflow_manager.py
      touch Simulation_check/${i}.done
    else
      echo "ERROR: Simulation failed with exit code \$sim_rc"

      # Leave start_Simulations_${i}.txt in place for debugging
      # Communicate failure (so workflow_manager can catch it)
      touch Simulation_check/${i}.fail
    fi

  fi

done
EOF

    sleep 0.25
    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script

echo "trainSim script ${i} initialized"

done