import sys
import json

def setup():
  
  


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

  setup()


if __name__ == "__main__":
  main()

