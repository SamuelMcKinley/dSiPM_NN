import os
import sys
import numpy as np

def find_npy_files(folder):
    """Find all .npy files in the folder (not including the summed_tensor output)."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".npy") and not f.startswith("summed_tensor_")
    ])

def main():
    if len(sys.argv) < 3:
        print("❌ Usage: python combine_tensors.py <folder_with_npy_files> <spad_size>")
        sys.exit(1)

    input_folder = sys.argv[1]
    spad_size = sys.argv[2].strip()

    # Validate folder and spad format
    if not os.path.exists(input_folder):
        print(f"❌ Folder '{input_folder}' not found.")
        sys.exit(1)
    if not os.path.isdir(input_folder):
        print(f"❌ '{input_folder}' is not a directory.")
        sys.exit(1)
    if not spad_size or "x" not in spad_size:
        print(f"❌ Invalid SPAD size '{spad_size}'. Expected format like '4000x4000'.")
        sys.exit(1)

    # Construct expected output folder and output filename
    output_folder_name = f"summed_tensor_{spad_size}_folder"
    output_dir = os.path.join(input_folder, output_folder_name)
    output_path = os.path.join(output_dir, f"summed_tensor_{spad_size}.npy")

    os.makedirs(output_dir, exist_ok=True)

    # Gather all .npy files in input folder
    all_npy_files = find_npy_files(input_folder)

    if not all_npy_files:
        print("❌ No .npy tensor files found in the folder.")
        sys.exit(1)

    combined_tensor = None
    loaded_count = 0

    # If the summed folder exists and contains a tensor, include it too
    existing_sum_path = os.path.join(output_dir, f"summed_tensor_{spad_size}.npy")
    if os.path.exists(existing_sum_path):
        try:
            print(f"📂 Found existing summed tensor at {existing_sum_path}. Including it in the new sum.")
            combined_tensor = np.load(existing_sum_path)
            loaded_count += 1
        except Exception as e:
            print(f"⚠️ Could not load existing summed tensor: {e}")

    # Sum all other tensors in input folder
    for npy_file in all_npy_files:
        try:
            tensor = np.load(npy_file)
        except Exception as e:
            print(f"⚠️ Could not load {npy_file}: {e}")
            continue

        if combined_tensor is None:
            combined_tensor = np.zeros_like(tensor)

        if combined_tensor.shape != tensor.shape:
            print(f"⚠️ Skipping incompatible tensor shape in {npy_file}: {tensor.shape}")
            continue

        combined_tensor += tensor
        loaded_count += 1

    if combined_tensor is None:
        print("❌ No valid tensors were loaded or combined.")
        sys.exit(1)

    np.save(output_path, combined_tensor)
    print(f"✅ Combined {loaded_count} tensors total.")
    print(f"📦 Saved summed tensor to {output_path}")

if __name__ == "__main__":
    main()