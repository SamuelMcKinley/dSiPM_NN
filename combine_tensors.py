import os
import sys
import numpy as np

def find_npy_files(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".npy") and f != "summed_tensor.npy"
    ])

def main():
    if len(sys.argv) < 3:
        print("❌ Usage: python combine_tensors.py <folder_with_npy_files> <output_folder_name>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder_name = sys.argv[2]

    all_npy_files = find_npy_files(input_folder)
    if not all_npy_files:
        print("❌ No .npy tensor files found in the folder.")
        sys.exit(1)

    output_dir = os.path.join(input_folder, output_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "summed_tensor.npy")

    combined_tensor = None
    loaded_count = 0

    # Load existing summed tensor if it exists
    if os.path.exists(output_path):
        print(f"📂 Found existing summed tensor at {output_path}. Adding to it.")
        combined_tensor = np.load(output_path)

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
        print("❌ No valid tensors were loaded.")
        sys.exit(1)

    np.save(output_path, combined_tensor)
    print(f"✅ Combined {loaded_count} tensors.")
    print(f"📦 Saved summed tensor to {output_path}")

if __name__ == "__main__":
    main()
