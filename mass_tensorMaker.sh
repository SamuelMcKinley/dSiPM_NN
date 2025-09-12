#!/bin/bash

cd jobs
Group_Size=$1

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
#SBATCH --mem-per-cpu=8G

# Replace with your environment setup
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:\$PATH
echo "Environment loaded."

cd ..
home_dir=\$PWD
temp_dir=/lustre/scratch/\$USER/dSiPM_NN/
cd \${temp_dir}

while squeue -u "\$USER" | grep -q "masterT"; do

  if [ ! -f "start_tensorMaker_${i}.txt" ]; then
    sleep 5
  else
    . start_tensorMaker_${i}.txt
    echo "Initialized for run ${i} at particle \$particle, energy \$energy, group \$group, SPAD Size \$SPAD_Size"

    python3 -u \${home_dir}/tensorMaker.py \
    mc_Simulation_run0_${i}_Test_1evt_\${particle}_\${energy}_\${energy}.root \
    \${energy} \
    tensor_\${group}_${i}_\${particle}_\${energy} \
    \${SPAD_Size}

    rm -rf mc_Simulation_run0_${i}_Test_1evt_\${particle}_\${energy}_\${energy}.root

    mv tensor_\${group}_${i}_\${particle}_\${energy}/npy/* tensfold/tensor_${i}_\${particle}_\${energy}_\${group}_\${SPAD_Size}.npy
    rm -rf tensor_\${group}_${i}_\${particle}_\${energy}

    echo "DONE"

    rm -rf start_tensorMaker_${i}.txt

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
