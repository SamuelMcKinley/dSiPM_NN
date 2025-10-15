#!/bin/bash

cd batch_jobs
Group_Size=$1

# Make the amount of scripts as there are groups
for ((i = 0; i < Group_Size; i++)); do

gen_script() {
    local script_name="mass_tensorMaker_${i}.sh"

    cat << EOF > "mass_tensorMaker_${i}.sh"
#!/bin/bash
#SBATCH -J "mass_tensorMaker_${i}.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 1
#SBATCH --mem-per-cpu=16G

# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."

cd ..
home_dir=\$PWD
temp_dir=/lustre/scratch/\$USER/dSiPM_NN/
cd \${temp_dir}

# Loop as long as the master script is running
while squeue -u "\$USER" | grep -q "masterT"; do

  # Look for text file used to communicate between masterTrain.sh and mass_tensorMaker.sh
  # Script will only run when file is found
  if [ ! -f "start_tensorMaker_${i}.txt" ]; then
    sleep 1
  else
    echo "Found start_tensorMaker_${i}.txt"

    # Copy over variables: ${particle}, ${energy}, ${group}, ${SPAD_Size}
    . start_tensorMaker_${i}.txt

    # Make time slices of xy projected photon tensors from the .root simulation files. Args: <simulation file> <energy> <output tensor folder name> <SPAD Size>
    python3 -u \${home_dir}/tensorMaker.py \
    mc_Simulation_run0_${i}_Test_1evt_\${particle}_\${energy}_\${energy}.root \
    \${energy} \
    tensor_\${group}_${i}_\${particle}_\${energy} \
    \${SPAD_Size}

    # Delete simulation file
    rm -rf mc_Simulation_run0_${i}_Test_1evt_\${particle}_\${energy}_\${energy}.root

    mv tensor_\${group}_${i}_\${particle}_\${energy}/npy/* tensfold/tensor_${i}_\${particle}_\${energy}_\${group}_\${SPAD_Size}.npy
    rm -rf tensor_\${group}_${i}_\${particle}_\${energy}

    echo "Tensor making complete for particle \${particle}, \${energy} GeV, group \${group}, SPAD_Size \${SPAD_Size}."

    # Remove communication text file
    rm -rf start_tensorMaker_${i}.txt

    # Communicate to masterTrain.sh that the tensor making is finished
    touch tensorMaker_check/${i}.done
  fi

done

EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

gen_script

echo "Tensor script ${i} started"

done