import os
import sys
import glob
import subprocess

# Pass variables from workflow_manager.py
group_size = int(sys.argv[1])
sim_check_dir = sys.argv[2]

# Pick up all of the names of the completed files
done_files = glob.glob(os.path.join(sim_check_dir, "*.done"))

def missing_done_indices(done_files, group_size):
    present = set()

    for path in done_files:
        base = os.path.basename(path)
        stem, ext = os.path.splitext(base)
        if ext != ".done":
            continue
        if stem.isdigit():
            present.add(int(stem))
    
    missing = [i for i in range(group_size) if i not in present]
    return missing

def main():
    if len(done_files) == group_size:
        print("Error: amount of done files matches group_size. Simulation_Failsafe.py should not have ran.")
        sys.exit(1)
    else:
        missing = missing_done_indices(done_files, group_size)

    print("The following events have timed out:")
    for i in missing:
        print(f"{i} / {group_size}")
    print("Restarting events...")
    for i in missing:
        job_name = f"Simulations_{i}"
        script_name = f"batch_jobs/Simulations_{i}.sh"
        subprocess.run(["scancel", "--name", job_name], check=False)
        subprocess.run(["sbatch", script_name], check=True)
        print(f"Restarted event {i}.")


if __name__ == "__main__":
    main()