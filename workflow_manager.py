import os
import sys
import json
import subprocess
import numpy as np
import random
import shutil
import glob
import time

home_dir = os.getcwd()
temp_dir = f"/lustre/scratch/{os.getenv('USER')}/dSiPM_NN"

def dir_setup():
  # Create scratch directories used for temporary file management
  if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
  os.makedirs(temp_dir)
  os.makedirs(f"{temp_dir}/Simulation_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/tensorMaking_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/NNTraining_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/tensfold", exist_ok=True)

def initialize_scripts(group_size):
  # Start running scripts
  subprocess.run(["./Simulations.sh", str(group_size)], check=True)
  subprocess.run(["./trainNN.sh"], check=True)
  subprocess.run(["./mass_tensorMaker.sh", str(group_size)], check=True)

def start_simulations(particle, energy, count):
  print("Starting simulations")
  # Create file with variables to initialize in bash script
  sim_filename = f"start_Simulations_{count}.txt"
  with open(sim_filename, "w") as f:
    f.write(f"particle={particle}\n")
    f.write(f"energy={energy}\n")
  shutil.move(sim_filename, f"{temp_dir}/")

def wait_for_simulation(group_size):
  sim_check_dir = os.path.join(temp_dir, "Simulation_check")

  print("Waiting for simulation jobs to finish ...")

  while True:
    # Check for all .done communication files
    s_done_files = glob.glob(os.path.join(sim_check_dir, "*.done"))
    s_done_count = len(s_done_files)

    if s_done_count == group_size:
      for f in s_done_files:
        os.remove(f)
      print("Simulation jobs finished")
      break
    time.sleep(0.5)

def start_tensorMaking(particle, energy, group, spad_size, count):
  print("Starting tensor making")
  # Create file with variables to initialize in bash script
  tens_filename = f"start_tensorMaker_{count}.txt"
  with open(tens_filename, "w") as f:
    f.write(f"particle={particle}\n")
    f.write(f"energy={energy}\n")
    f.write(f"group={group}\n")
    f.write(f"SPAD_Size={spad_size}\n")
  shutil.move(tens_filename, f"{temp_dir}/")

def wait_for_tensorMaking(group_size):
  tens_check_dir = os.path.join(temp_dir, "tensorMaking_check")

  print("Waiting for tensor making jobs to finish ...")

  while True:
    # Check for all .done communication files
    t_done_files = glob.glob(os.path.join(tens_check_dir, "*.done"))
    t_done_count = len(t_done_files)

    if t_done_count == group_size:
      for f in t_done_files:
        os.remove(f)
      print("Tensor making jobs finished")
      break
    time.sleep(0.5)

def start_NNTraining(spad_size, energy):
  print("Starting NN training")
  # Create file with variables to initialize in bash script
  NN_filename = f"start_training.txt"
  with open(NN_filename, "w") as f:
    f.write(f"SPAD_Size={spad_size}\n")
    f.write(f"energy={energy}\n")
  shutil.move(NN_filename, f"{temp_dir}/")

def wait_for_NNTraining(group_size):
  NN_check_dir = os.path.join(temp_dir, "NNTraining_check")

  print("Waiting for NN training jobs to finish ...")

  while True:
    # Check for all .done communication files
    n_done_files = glob.glob(os.path.join(NN_check_dir, "*.done"))
    n_done_count = len(n_done_files)

    # Built in logic in case want to expand to parallel training. Can change 1 to group_size
    if n_done_count == 1:
      for f in n_done_files:
        os.remove(f)
      print("NN training job finished")
      break
    time.sleep(0.5)

# Analysis A runs every group
def Analysis_A(group_size):
    wait_for_NNTraining(group_size)

# Analysis B runs once per SPAD Size / model
def Analysis_B(spad_size):
  print("Making NN analysis plots")
  subprocess.run(["python3", "-u", "plot_loss.py", f"NNTraining/{spad_size}_model/NN_model_{spad_size}"], check=True)
  shutil.move("plots", f"Training_Outputs/plots_{spad_size}")

def Analysis_C():
  subprocess.run(["python3", "-u", "clean_csv.py"])
  subprocess.run(["python3", "-u", "plot_photon_tracking.py", "photon_tracking_combined.csv"], check=True)
  shutil.move("photon_tracking_plots", "Training_Outputs/")

def main():

  if os.path.exists("Training_Outputs"):
    shutil.rmtree("Training_Outputs")
  os.makedirs("Training_Outputs/")

  config_file = sys.argv[1]

  with open(config_file, "r") as f:
    config = json.load(f)

  # Pull Parameters from JSON file
  energy_min   = int(config["energy_min"])
  energy_max   = int(config["energy_max"])
  energy_step  = int(config["energy_step"])
  Events_per_energy = int(config["total_events"])
  group_size   = int(config["group_size"])
  particle     = config["particle"]
  spad_sizes   = config["spad_sizes"]

  print("\n--- Workflow Parameters ---")
  print(f"Energy: {energy_min} => {energy_max} step {energy_step} GeV")
  print(f"Total events: {Events_per_energy} (group size {group_size})")
  print(f"Particle: {particle}")
  print(f"SPAD sizes: {spad_sizes}\n")

  dir_setup()

  initialize_scripts(group_size)

  energies = np.arange(energy_min, energy_max + energy_step, energy_step)
  specific_group_size = group_size // len(energies)

  groups = (Events_per_energy * len(energies)) // group_size
  print(f"{groups} groups total.")

  expanded_energies = np.repeat(energies, specific_group_size)

  # Master loops to run all subprocesses
  for spad_size in spad_sizes:
    for group in range(0, groups):
      print(f"Starting group {group + 1}/{groups} for SPAD {spad_size}")
      #Shuffles energies for better NN results
      np.random.shuffle(expanded_energies)
      
      # Loop GEANT4 Simulations

      count = -1
      for energy in expanded_energies:
        count += 1
        start_simulations(particle, energy, count)
      wait_for_simulation(group_size)

      # Function checks for NN training from previous iteration to be complete
      # Allows NN training overlap with GEANT 4 Sim
      if group != 0:
        Analysis_A(group_size)

      # Loop tensor making
      count = -1
      for energy in expanded_energies:
        count += 1
        start_tensorMaking(particle, energy, group, spad_size, count)
      wait_for_tensorMaking(group_size)

      # Run Cummulative NN training
      start_NNTraining(spad_size, energy)

      # Run Analysis
      if group == (groups - 1):
        Analysis_A(group_size)

    Analysis_B(spad_size)
  Analysis_C()


if __name__ == "__main__":
  main()
