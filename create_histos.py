import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.colors import ListedColormap, BoundaryNorm

def get_custom_colormap(global_max):
    """
    Creates a colormap with white for 0, and a smooth blue-to-yellow gradient for 1 to global_max.
    """
    n_bins = min(int(global_max), 255)  # Max 255 bins
    white = np.array([[1.0, 1.0, 1.0, 1.0]])  # white for 0

    # Create gradient: blue → cyan → green → yellow (no red or purple)
    blue_to_yellow = np.column_stack((
        np.linspace(0.0, 1.0, n_bins),          # R (0 to 1)
        np.linspace(0.0, 1.0, n_bins),          # G (0 to 1)
        np.linspace(1.0, 0.0, n_bins),          # B (1 to 0)
        np.ones(n_bins)                         # Alpha
    ))

    colors = np.vstack((white, blue_to_yellow))
    cmap = ListedColormap(colors)
    boundaries = np.arange(-0.5, n_bins + 1.5, 1)
    norm = BoundaryNorm(boundaries, ncolors=len(colors), clip=True)
    return cmap, norm

def plot_slice(hist2d, time_range, spad_size, output_dir, idx, cmap, norm):
    entries = np.sum(hist2d)
    
    x_extent = [-80, 80]
    y_extent = [-80, 80]

    y_indices, x_indices = np.meshgrid(np.arange(hist2d.shape[1]), np.arange(hist2d.shape[0]))
    x_centers = np.linspace(x_extent[0], x_extent[1], hist2d.shape[0])
    y_centers = np.linspace(y_extent[0], y_extent[1], hist2d.shape[1])

    X = np.tile(x_centers[:, None], (1, hist2d.shape[1]))
    Y = np.tile(y_centers[None, :], (hist2d.shape[0], 1))

    mean_x = np.average(X, weights=hist2d)
    mean_y = np.average(Y, weights=hist2d)
    std_x = np.sqrt(np.average((X - mean_x)**2, weights=hist2d))
    std_y = np.sqrt(np.average((Y - mean_y)**2, weights=hist2d))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(hist2d, extent=y_extent + x_extent, origin='lower', cmap=cmap, norm=norm, aspect='equal')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Photon Count")

    ax.set_title(f"Photon Hits from {time_range[0]} to {time_range[1]} ns, Channel Size {spad_size}")
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("x [mm]")

    stats = (
        f"Entries = {int(entries)}\n"
        f"Mean x = {mean_x:.2f}\n"
        f"Mean y = {mean_y:.2f}\n"
        f"Std Dev x = {std_x:.2f}\n"
        f"Std Dev y = {std_y:.2f}"
    )
    ax.text(0.03, 0.97, stats, transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top', horizontalalignment='left')

    filename = f"nPhotons_xy_t{time_range[0]}to{time_range[1]}_ch{spad_size}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot time-sliced photon tensors.")
    parser.add_argument("tensor_file", help="Path to .npy file containing combined tensor")
    parser.add_argument("spad_size", help="SPAD size string (e.g. 70x70) for labeling")
    args = parser.parse_args()

    time_slices = [(0, 9), (9, 9.5), (9.5, 10), (10, 15), (15, 40)]

    tensor = np.load(args.tensor_file)
    output_dir = os.path.dirname(os.path.abspath(args.tensor_file))
    print(f"Loaded tensor with shape {tensor.shape} from {args.tensor_file}")

    if tensor.shape[0] != len(time_slices):
        raise ValueError("Tensor time slices do not match expected time slice count (5).")

    global_max = np.max(tensor)
    print(f"Global max photon count: {global_max}")
    cmap, norm = get_custom_colormap(global_max)

    for idx, time_range in enumerate(time_slices):
        print(f"Plotting time slice {time_range}...")
        plot_slice(tensor[idx], time_range, args.spad_size, output_dir, idx, cmap, norm)

if __name__ == "__main__":
    main()
