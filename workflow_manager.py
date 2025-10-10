import os
import sys
import json
import subprocess
import numpy as np
import random

home_dir = os.getcwd()
temp_dir = f"/lustre/scratch/{os.getenv('USER')}/dSiPM_NN"

def dir_setup():
  # Create scratch directories used for temporary file management
  os.makedirs(f"{temp_dir}", exist_ok=True)
  os.makedirs(f"{temp_dir}/Simulation_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/tensorMaking_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/NNTraining_check", exist_ok=True)
  os.makedirs(f"{temp_dir}/tensfold", exist_ok=True)

def check_event_count(total_events, group_size):
  if (total_events % group_size != 0):
    print ("Events not divisible by batch size. Please re-run script with acceptable parameters")
    sys.exit(1)  

def initialize_scripts(group_size):
  # Start running scripts
  subprocess.run(["./Simulations.sh", group_size], check=True)
  subprocess.run(["./trainNN.sh"], check=True)
  subprocess.run(["./mass_tensorMaker.sh", group_size], check=True)

def start_simulations(particle, energy):
  for 





def main():

  config_file = sys.argv[1]

  with open(config_file, "r") as f:
    config = json.load(f)

  # Pull Parameters from JSON file
  energy_min   = config["energy_min"]
  energy_max   = config["energy_max"]
  energy_step  = config["energy_step"]
  total_events = config["total_events"]
  group_size   = config["group_size"]
  particle     = config["particle"]
  spad_sizes   = config["spad_sizes"]

  print("\n--- Workflow Parameters ---")
  print(f"Energy: {energy_min} → {energy_max} step {energy_step} GeV")
  print(f"Total events: {total_events} (group size {group_size})")
  print(f"Particle: {particle}")
  print(f"SPAD sizes: {spad_sizes}\n")

  dir_setup()

  check_event_count(total_events, group_size)

  groups = total_events / group_size

  initialize_scripts(group_size)

  energies = list(range(energy_min, energy_max + energy_step, energy_step))

  # Master loops to run all subprocesses
  for spad_size in spad_sizes:
    for group in range(0, groups):
      #Shuffles energies for better NN results
      random.shuffle(energies)
      for energy in energies:
        start_simulations(particle, energy)




        


if __name__ == "__main__":
  main()