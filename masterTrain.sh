#!/bin/bash

cd jobs
mkdir -p LOGDIR
rm -rf LOGDIR/*

gen_script() {
    local script_name="masterTrain.sh"

    cat << 'EOF' > "masterTrain.sh"
#!/bin/bash
#SBATCH -J "masterTrain.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 2
#SBATCH --mem-per-cpu=16G

# Edit Parameters here:
particle="pi+"
# Energy in GeV
Lower_E=5
Upper_E=10
Step_Size=5
# Amount of Events
Events=10
# Group Size
Group_Size=5
#Expand array as needed
SPAD_Sizes=("4000x4000" "2000x2000")

temp_dir=/lustre/scratch/$USER/dSiPM_NN
mkdir -p ${temp_dir}
mkdir -p ${temp_dir}/Simulation_check
mkdir -p ${temp_dir}/tensorMaker_check
mkdir -p ${temp_dir}/NNTraining_check
mkdir -p ${temp_dir}/tensfold

cd ..

home_dir=$PWD

echo "Training neural network with particle ${particle} from ${Lower_E}GeV to \
${Upper_E}GeV in steps of ${Step_Size}GeV. ${Events} events per energy."

# Check if Events divisible by 100
if (( Events % Group_Size == 0 )); then
    group=$(( Events / Group_Size ))
else
    echo "Events not divisible by ${Group_Size}. Code runs in batches of ${Group_Size}. Please adjust"
    exit 0
fi

# Start the files
./Simulations.sh ${Group_Size}
./trainNN.sh
./mass_tensorMaker.sh ${Group_Size}


# Loop over SPAD Sizes, Energies, and groups of 100
for s in "${SPAD_Sizes[@]}"; do
for (( i=${Lower_E}; i<=${Upper_E}; i+=${Step_Size} )); do
for j in $(seq 1 $group); do

# If it's the first event, start the simulations and tensor making right away
if [[ "$j" == 1 ]]; then

echo "Starting first batch of simulations"

for ((n = 0; n < Group_Size; n++)); do
# Make text file to communicate with Simulation.sh
cat > start_Simulations_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${j}
SPAD_Size=${s}
EOT

mv start_Simulations_${n}.txt ${temp_dir}/

done

echo "Simulation jobs started"

# Wait for Simulation to finish
while true; do
  done_count=$(ls ${temp_dir}/Simulation_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq "${Group_Size}" ]; then
    rm -rf ${temp_dir}/Simulation_check/*.done
    break
  fi
    sleep 1
done

echo "Simulation jobs finished"

echo "Making tensors"

# Now run code to make tensors

####################### run mass_tensorMaker.sh
for ((n = 0; n < Group_Size; n++)); do
# Make text file to communicate with Simulation.sh
cat > start_tensorMaker_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${j}
SPAD_Size=${s}
EOT

mv start_tensorMaker_${n}.txt ${temp_dir}/

done
#######################
# Will output tensor_${j}_${number}_${particle}_${i}.npy

echo "Tensor jobs started"


# Wait for tensor making to finish
while true; do
  done_count=$(ls ${temp_dir}/tensorMaker_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq "${Group_Size}" ]; then
    rm -rf ${temp_dir}/tensorMaker_check/*.done
    break
  fi
    sleep 1
done

echo "Tensor jobs finished"



fi




echo "Starting training scripts"

#Copying and starting trainNN.sh through text file communication
cat > start_training.txt <<EOT
particle=${particle}
energy=${i}
group=${j}
SPAD_Size=${s}
EOT

mv start_training.txt ${temp_dir}/


if [[ "$j" == "${group}" ]]; then

while true; do
  done_count=$(ls ${temp_dir}/NNTraining_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq 1 ]; then
    rm -rf ${temp_dir}/NNTraining_check/*.done
    break
  fi
    sleep 1
done


fi




if [[ "$j" != "${group}" ]]; then

echo "Starting next simulation batch"

next_group=$((j + 1))

for ((n = 0; n < Group_Size; n++)); do
# Make text file to communicate with Simulation.sh
cat > start_Simulations_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${next_group}
SPAD_Size=${s}
EOT

mv start_Simulations_${n}.txt ${temp_dir}/

done

echo "trainSim.sh started"

while true; do
  done_count=$(ls ${temp_dir}/Simulation_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq "${Group_Size}" ]; then
    rm -rf ${temp_dir}/Simulation_check/*.done
    break
  fi
    sleep 1
done

echo "Simulation jobs finished"

while true; do
  done_count=$(ls ${temp_dir}/NNTraining_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq 1 ]; then
    rm -rf ${temp_dir}/NNTraining_check/*.done
    break
  fi
    sleep 1
done

echo "Training jobs finished"

echo "Making tensors"

# Now run code to make tensors

####################### run mass_tensorMaker.sh
for ((n = 0; n < Group_Size; n++)); do
# Make text file to communicate with Simulation.sh
cat > start_tensorMaker_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${next_group}
SPAD_Size=${s}
EOT

mv start_tensorMaker_${n}.txt ${temp_dir}/

done
#######################
# Will output tensor_${j}_${number}_${particle}_${i}.npy

echo "Tensor jobs started"


# Wait for tensor making to finish
while true; do
  done_count=$(ls ${temp_dir}/tensorMaker_check/*.done 2>/dev/null | wc -l)
  if [ "$done_count" -eq "${Group_Size}" ]; then
    rm -rf ${temp_dir}/tensorMaker_check/*.done
    break
  fi
    sleep 1
done

echo "Tensor jobs finished"


fi

rm -rf ${temp_dir}/NNTraining_check/*.done

echo "Group ${j} in energy ${i} for SPAD size ${s} complete"

done

cd NNTraining/${s}_model
python3 -u plot_loss.py NN_model_${s}/loss_history_${i}.txt
cd ../..

echo "Energy ${i} for SPAD size ${s} complete"

done

cd output_${s}

# Creating histograms
echo "Creating histogram for ${s} SPADs"
python3 -u ../create_histos.py \
summed_tensor_${s}/summed_tensor.npy \
${s}

cd ${home_dir}

echo "SPAD size ${s} complete"

done

echo "masterTrain.sh finished"

#####################################################
EOF

    chmod +x "$script_name"
    sbatch "$script_name"
}

echo "Script generated"

gen_script
