#!/bin/bash
#SBATCH -J "masterTrain.sh"
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -o LOGDIR/%x.%j.out
#SBATCH -e LOGDIR/%x.%j.err
#SBATCH -p nocona
#SBATCH -c 2
#SBATCH --mem-per-cpu=16G

################ Edit Parameters here: ################
particle="pi+"
# Energy in GeV
Lower_Energy_Bound=5
Upper_Energy_Bound=10
Energy_Step_Size=5
# Amount of Events
Events=10
# Batch Size (Keep Events divisible by Batch Size. Recommended <= 500 jobs maximum)
Batch_Size=5
#Expand array as needed
SPAD_Sizes=("4000x4000" "2000x2000")


# Load environment needed for python imports
echo "Loading environment ..."
export PATH=~/miniconda3/envs/base/bin:$PATH
echo "Environment loaded."

temp_dir=/lustre/scratch/$USER/dSiPM_NN
mkdir -p ${temp_dir}
mkdir -p ${temp_dir}/Simulation_check
mkdir -p ${temp_dir}/tensorMaker_check
mkdir -p ${temp_dir}/NNTraining_check
mkdir -p ${temp_dir}/tensfold
cd ..
home_dir=$PWD

echo "Training neural network with particle ${particle} from ${Lower_Energy_Bound}GeV to \
${Upper_Energy_Bound}GeV in steps of ${Energy_Step_Size}GeV. ${Events} events per energy."


# Check if Events divisible by specified batch size
if (( Events % Batch_Size == 0 )); then
    group=$(( Events / Batch_Size ))
else
    echo "Events not divisible by ${Batch_Size}. Code runs in batches of ${Batch_Size}. Please adjust"
    exit 0
fi


# Start the main scripts
./Simulations.sh ${Batch_Size}
./trainNN.sh
./mass_tensorMaker.sh ${Batch_Size}


# Loop over SPAD Sizes, Energies, and Groups (group = Total Events / Batch Size)
for s in "${SPAD_Sizes[@]}"; do
  for (( i=${Lower_Energy_Bound}; i<=${Upper_Energy_Bound}; i+=${Energy_Step_Size} )); do
    for j in $(seq 1 $group); do


      # If it's the first event, start the simulations and tensor making right away
      if [[ "$j" == 1 ]]; then

        echo "Starting first training batch"


        # Make text files to communicate with Simulation.sh and start first batch
        for ((n = 0; n < Batch_Size; n++)); do

          cat > start_Simulations_${n}.txt <<EOT
particle=${particle}
energy=${i}
EOT

          mv start_Simulations_${n}.txt ${temp_dir}/

        done

        echo "Simulation jobs started"


        # Wait for simulations to finish
        while true; do
          done_count=$(ls ${temp_dir}/Simulation_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq "${Batch_Size}" ]; then
            rm -rf ${temp_dir}/Simulation_check/*.done
            echo "Simulation jobs finished"
            break
          fi
            sleep 1
        done



        # Make text files to communicate with mass_tensorMaker.sh and start first batch
        for ((n = 0; n < Batch_Size; n++)); do

          cat > start_tensorMaker_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${j}
SPAD_Size=${s}
EOT

          mv start_tensorMaker_${n}.txt ${temp_dir}/

        done

        echo "Tensor jobs started"


        # Wait for tensor making to finish
        while true; do
          done_count=$(ls ${temp_dir}/tensorMaker_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq "${Batch_Size}" ]; then
            rm -rf ${temp_dir}/tensorMaker_check/*.done
            echo "Tensor jobs finished"
            break
          fi
            sleep 1
        done




      fi




      # Make text files to communicate with trainNN.sh and start training
      cat > start_training.txt <<EOT
SPAD_Size=${s}
energy=${i}
EOT

      mv start_training.txt ${temp_dir}/

      echo "Training job started"

      if [[ "$j" == "${group}" ]]; then

        # Waiting for training to finish
        while true; do
          done_count=$(ls ${temp_dir}/NNTraining_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq 1 ]; then
            rm -rf ${temp_dir}/NNTraining_check/*.done
            echo "Training job finished"
            break
          fi
            sleep 1
        done


      fi




      if [[ "$j" != "${group}" ]]; then

        echo "Starting next simulation batch"

        next_group=$((j + 1))

        for ((n = 0; n < Batch_Size; n++)); do
          # Make text files to communicate with Simulations.sh and start next batch
          cat > start_Simulations_${n}.txt <<EOT
particle=${particle}
energy=${i}
EOT

          mv start_Simulations_${n}.txt ${temp_dir}/

        done

        echo "Simulation jobs started"


        # Waiting for simulations to finish
        while true; do
          done_count=$(ls ${temp_dir}/Simulation_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq "${Batch_Size}" ]; then
            rm -rf ${temp_dir}/Simulation_check/*.done
            echo "Simulation jobs finished"
            break
          fi
            sleep 1
        done


        # Waiting for training to finish
        while true; do
          done_count=$(ls ${temp_dir}/NNTraining_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq 1 ]; then
            rm -rf ${temp_dir}/NNTraining_check/*.done
            echo "Training job finished"
            break
          fi
            sleep 1
        done



        for ((n = 0; n < Batch_Size; n++)); do
          # Make text files to communicate with mass_tensorMaker.sh and start next batch
          cat > start_tensorMaker_${n}.txt <<EOT
particle=${particle}
energy=${i}
group=${next_group}
SPAD_Size=${s}
EOT

          mv start_tensorMaker_${n}.txt ${temp_dir}/

        done

        echo "Tensor jobs started"


        # Wait for tensor making to finish
        while true; do
          done_count=$(ls ${temp_dir}/tensorMaker_check/*.done 2>/dev/null | wc -l)
          if [ "$done_count" -eq "${Batch_Size}" ]; then
            rm -rf ${temp_dir}/tensorMaker_check/*.done
            echo "Tensor jobs finished"
            break
          fi
            sleep 1
        done



      fi


      echo "Group ${j} in energy ${i} for SPAD size ${s} complete"

    done


    # Plot the losses throughout NN epochs
    echo "Plotting losses"
    cd NNTraining/${s}_model
    python3 -u plot_loss.py NN_model_${s}/loss_history_${i}.txt
    cd ../..

    # Prepare nPhoton vs Energy plots
    echo "Making csv file for plotting"
    cd output_${s}
    python3 -u ../n_vs_e_csv.py summed_tensor_${i}_${s}/summed_tensor.npy ${i}
    cp summed_tensor_${i}_${s}/summed_tensor.npy summed_tensor_${i}.npy
    cd ..


    echo "Energy ${i} for SPAD size ${s} complete"

  done

  # Sum all energies of tensors together
  cd output_${s}
  echo "Summing tensors"
  python3 -u ../combine_tensors.py . summed_tensor_${s}
  mv *.npy summed_tensor_${s}/
  cd ..

  # Plots nPhoton vs Energy
  echo "Plotting nPhoton vs Energy"
  cd output_${s}
  python3 -u ../n_vs_e_plotting.py photon_counts.csv ${s}
  cd ..

  cd output_${s}/summed_tensor_${s}
  # Creating time-sliced xy projected photon histograms
  echo "Creating histogram for ${s} SPADs"
  python3 -u ../../create_histos.py \
  summed_tensor.npy \
  ${s}
  cd ${home_dir}



  echo "SPAD size ${s} complete"

done


# Moving and cleaning outputs
mkdir -p Training_Outputs/
for s in "${SPAD_Sizes[@]}"; do
  mv output_${s}/summed_tensor_${s}/nPhotons* Training_Outputs/
  mv output_${s}/nPhotons* Training_Outputs/
  mv output_${s}/summed_tensor_${s}/summed_tensor.npy Training_Outputs/summed_tensor_${s}.npy
  rm -rf output_${s}
  mkdir -p Training_Outputs/Loss_Plots_${s}
  mv NNTraining/${s}_model/loss_plots/* Training_Outputs/Loss_Plots_${s}/
done


echo "masterTrain.sh finished"

