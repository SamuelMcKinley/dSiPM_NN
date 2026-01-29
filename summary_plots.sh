#!/bin/bash

Events_per_energy=$1
SPAD_Sizes=("${@:2}")
Analysis_validations=$((Events_per_energy / 10))

python3 -u clean_csv.py

python3 -u plot_photon_tracking.py photon_tracking_combined.csv "${Events_per_energy}"

mv photon_tracking_plots Training_Outputs/

for SPAD_Size in "${SPAD_Sizes[@]}"; do

    python3 -u make_resolution.py --csv NNTraining/${SPAD_Size}_model/NN_model_${SPAD_Size}/val_predictions_all_epochs.csv \
    --out ${SPAD_Size}_Eres.png --n_per_energy "${Analysis_validations}"

    mv ${SPAD_Size}_Eres.png Training_Outputs/
done