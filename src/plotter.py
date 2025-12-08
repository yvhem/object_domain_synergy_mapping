import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Radius, Position
    h_radius = df['HumanRadius']
    r_radius = df['RobotRadius']
    h_pos = df[['hPos_x', 'hPos_y', 'hPos_z']].values
    r_pos = df[['rPos_x', 'rPos_y', 'rPos_z']].values

    # Energy and Scaling Factor
    h_energy = df['HumanEnergy']
    r_energy = df['RobotEnergy']
    k = df['ScalingFactor'].iloc[0]
    time = df['Time']
    return df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, k

def compute_metrics(h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_energy, aligned_r_energy, k):
    # Radius
    scaled_h_radius = h_radius * k
    rad_error = np.abs(aligned_r_radius - scaled_h_radius)

    # Position
    h_start_pos = h_pos[0]
    r_start_pos = aligned_r_pos[0]
    h_delta_vec = h_pos - h_start_pos
    r_delta_vec = aligned_r_pos - r_start_pos
    h_move_mag = np.linalg.norm(h_delta_vec, axis=1)
    r_move_mag = np.linalg.norm(r_delta_vec, axis=1)
    pos_error_var = np.abs(h_move_mag - r_move_mag)

    # Energy
    exp_r_energy = h_energy
    energy_delta = np.abs(aligned_r_energy - exp_r_energy)


    return (scaled_h_radius, aligned_r_radius, rad_error), \
        (h_move_mag, r_move_mag, pos_error_var), \
        (exp_r_energy, aligned_r_energy, energy_delta)


class ResultVisualizer:
    def __init__(self, time, radius_data, pos_data, energy_data, _fps):
        self.time = time
        self.fps = _fps

        # Styles
        sns.set_context("paper", font_scale=5.5)
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 26
        plt.rcParams['axes.titlesize'] = 32
        plt.rcParams['axes.labelsize'] = 28
        plt.rcParams['xtick.labelsize'] = 24
        plt.rcParams['ytick.labelsize'] = 24
        plt.rcParams['legend.fontsize'] = 24
        plt.rcParams['figure.titlesize'] = 34
        plt.rcParams['lines.linewidth'] = 4

        self.colors = {'h': '#1f77b4', 'r': '#ff7f0e', 'err_f': '#ff9999', 'err_l': '#d62728', 'pct': '#800080'}
        self.lw = 6.0

        # Initialize Figure
        self.fig, self.axes = plt.subplots(3, 1, figsize=(15, 25), sharex=True)
        self.axes_flat = self.axes.flatten()

        # Data grouping for iteration (Radius=0, Pos=1, Energy=2)
        self.plot_defs = [
            (self.axes_flat[0], "Radius Comparison", "Radius (m)", radius_data),
            (self.axes_flat[1], "Position Variation", "Displacement (m)", pos_data),
            (self.axes_flat[2], "Elastic Energy", "Energy (J)", energy_data)
        ]

        self.lines = []

        # Setup first 3 plots
        for ax, title, ylabel, data in self.plot_defs:
            self._setup_axis(ax, title, ylabel, data[0], data[1], data[2])

        # Manually add xlabel to bottom-left plot (Energy)
        self.axes_flat[2].set_xlabel('Time (s)', fontweight='bold', fontsize=28)
        self.axes_flat[2].tick_params(axis='x', which='major', labelsize=24)

        plt.tight_layout(pad=2.5)

    def _setup_axis(self, ax, title, ylabel, h_data, r_data, e_data):
        ax.set_title(title, loc='left', fontweight='bold', fontsize=32)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=24, length=6, width=2)
        ax.set_xlim(self.time.min(), self.time.max())

        # Calculate Y limits
        all_vals = np.concatenate([h_data, r_data, e_data])
        y_min, y_max = all_vals.min(), all_vals.max()
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)

        # Initialize Empty Lines
        l_err, = ax.plot([], [], color=self.colors['err_l'], linewidth=2, alpha=0.6)
        l_h, = ax.plot([], [], color=self.colors['h'], linestyle='--', linewidth=self.lw, label='Human')
        l_r, = ax.plot([], [], color=self.colors['r'], linewidth=self.lw, alpha=0.9, label='Robot')

        ax.legend(loc='best', framealpha=0.95, fontsize=24, borderpad=0.8)

        # Store references for updating
        self.lines.append({
            'ax': ax, 'l_h': l_h, 'l_r': l_r, 'l_err': l_err,
            'd_h': h_data, 'd_r': r_data, 'd_err': e_data
        })

    def update_plot(self, frame_idx):
        if frame_idx == -1:
            t_slice = self.time
            idx = len(self.time)
        else:
            t_slice = self.time.iloc[:frame_idx]
            idx = frame_idx

        for item in self.lines:
            item['l_h'].set_data(t_slice, item['d_h'][:idx])
            item['l_r'].set_data(t_slice, item['d_r'][:idx])
            item['l_err'].set_data(t_slice, item['d_err'][:idx])
            for collection in item['ax'].collections:
                collection.remove()
            item['ax'].fill_between(t_slice, 0, item['d_err'][:idx], color=self.colors['err_f'], alpha=0.4)

        return []

    def save_static(self, path):
        self.update_plot(-1)
        self.fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Static Graph saved to: {path}")

    def save_animation(self, path):
        print("Generating animation... (this may take a moment)")
        total_points = len(self.time)
        stride = max(1, total_points // 300)
        frames = range(1, total_points, stride)

        ani = FuncAnimation(self.fig, self.update_plot, frames=frames, interval=20, blit=False)

        pbar = tqdm(total=len(frames), desc="Rendering Frames", unit="frame")

        def progress_callback(current, total):
            pbar.update(1)

        try:
            ani.save(
                path,
                writer='ffmpeg',
                fps=self.fps,
                dpi=150,
                progress_callback=progress_callback
            )
            pbar.close()
            print(f"\nAnimation saved to: {path}")

        except Exception as e:
            pbar.close()
            print(f"\nFFmpeg error: {e}. Trying GIF...")
            try:
                gif_path = path.replace('.mp4', '.gif')
                ani.save(gif_path, writer='pillow', fps=60, dpi=100)
                print(f"GIF saved to: {gif_path}")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")

    def show(self):
        plt.show()


def main(csv_path, anim_fps, verbose, mode):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    try:
        df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, k = read_csv(csv_path)
    except Exception as e:
        print(f"CSV Reading Error: {e}")
        return

    radius_data, pos_data, energy_data = compute_metrics(
        h_radius, r_radius, h_pos, r_pos, h_energy, r_energy, k
    )

    if verbose:
        print("=" * 40 + "\n      GLOBAL RMSE SUMMARY       \n" + "=" * 40)
        print(f"Radius RMSE:       {np.sqrt(np.mean(radius_data[2] ** 2)):.5f} m")
        print(f"Pos Variation RMSE:{np.sqrt(np.mean(pos_data[2] ** 2)):.5f} m")
        print("=" * 40)

    viz = ResultVisualizer(time, radius_data, pos_data, energy_data, anim_fps)

    if mode in (1,3):
        save_path_static = os.path.splitext(csv_path)[0] + ".png"
        viz.save_static(save_path_static)

    if mode in (2,3):
        save_path_anim = os.path.splitext(csv_path)[0] + "_video.mp4"
        viz.save_animation(save_path_anim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV file path")
    parser.add_argument("--fps", help="Animation FPS", default=60)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print details")
    parser.add_argument( "--mode", type=int, choices=[1,2,3], default=1 , help="1. Plots; 2. Animation; 3. Both")

    args = parser.parse_args()

    main(args.path, args.fps, args.verbose, args.mode)
