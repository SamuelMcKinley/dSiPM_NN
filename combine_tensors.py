import os
import sys
import numpy as np


def find_npz_files(folder: str):
    """Find all .npz files in the folder (excluding anything inside summed_tensor_* folders)."""
    out = []
    for f in os.listdir(folder):
        if f.endswith(".npz") and f.startswith("tensor_"):
            out.append(os.path.join(folder, f))
    return sorted(out)


def load_real_tensor_from_npz(npz_path: str) -> np.ndarray:
    """
    Loads {x, lnN} from an event .npz and reconstructs the REAL hit-map:
      real_tensor = x * exp(lnN)
    where x was normalized by denom (kept photon count).
    """
    z = np.load(npz_path)
    if "x" not in z.files or "lnN" not in z.files:
        raise KeyError(f"{npz_path} missing required keys. Found keys: {z.files}")

    x = z["x"].astype(np.float32, copy=False)
    lnN = float(z["lnN"])
    denom = float(np.exp(lnN))  # back to photon count scale

    # Reconstruct original hit-map counts
    return (x * denom).astype(np.float32, copy=False)


def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_tensors.py <folder_with_npz_files> <spad_size>")
        sys.exit(1)

    input_folder = sys.argv[1]
    spad_size = sys.argv[2].strip()

    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' not found.")
        sys.exit(1)
    if not os.path.isdir(input_folder):
        print(f"'{input_folder}' is not a directory.")
        sys.exit(1)
    if not spad_size or "x" not in spad_size:
        print(f"Invalid SPAD size '{spad_size}'. Expected format like '4000x4000'.")
        sys.exit(1)

    output_folder_name = f"summed_tensor_{spad_size}_folder"
    output_dir = os.path.join(input_folder, output_folder_name)
    output_path = os.path.join(output_dir, f"summed_tensor_{spad_size}.npy")
    os.makedirs(output_dir, exist_ok=True)

    # Gather all .npz event tensors in input folder
    all_npz_files = find_npz_files(input_folder)
    if not all_npz_files:
        print("No .npz tensor files found in the folder.")
        sys.exit(1)

    combined_tensor = None
    loaded_count = 0

    # If an existing summed tensor exists, include it too
    if os.path.exists(output_path):
        try:
            print(f"Found existing summed tensor at {output_path}. Including it in the new sum.")
            combined_tensor = np.load(output_path).astype(np.float32, copy=False)
            loaded_count += 1
        except Exception as e:
            print(f"Could not load existing summed tensor: {e}")

    # Sum all event tensors (reconstructing real hit-map first)
    for npz_file in all_npz_files:
        try:
            tensor = load_real_tensor_from_npz(npz_file)
        except Exception as e:
            print(f"Could not load {npz_file}: {e}")
            continue

        if combined_tensor is None:
            combined_tensor = np.zeros_like(tensor, dtype=np.float32)

        if combined_tensor.shape != tensor.shape:
            print(f"Skipping incompatible tensor shape in {npz_file}: {tensor.shape}")
            continue

        combined_tensor += tensor
        loaded_count += 1

    if combined_tensor is None:
        print("No valid tensors were loaded or combined.")
        sys.exit(1)

    np.save(output_path, combined_tensor.astype(np.float32, copy=False))
    print(f"Combined {loaded_count} tensors total.")
    print(f"Saved summed tensor to {output_path}")


if __name__ == "__main__":
    main()