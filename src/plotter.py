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


def preprocess(h_radius, r_radius, r_energy, r_pos, verbose, enable_shift=True):
    shift_idx = 0

    if enable_shift:
        h_mid = (h_radius.max() + h_radius.min()) / 2.0
        r_mid = (r_radius.max() + r_radius.min()) / 2.0
        idx_mid_h = (np.abs(h_radius - h_mid)).argmin()
        idx_mid_r = (np.abs(r_radius - r_mid)).argmin()
        shift_idx = idx_mid_r - idx_mid_h
    elif verbose:
        print("Shifting disabled by user.")

    if verbose and enable_shift:
        print(f"Shift Applied: {shift_idx} frames")

    def shift_array(arr, shift):
        if shift == 0:
            return arr
        result = np.roll(arr, -shift, axis=0)
        if shift > 0:
            result[-shift:] = result[-shift - 1]
        elif shift < 0:
            result[:-shift] = result[-shift]
        return result

    aligned_r_radius = r_radius.shift(-shift_idx).ffill().bfill()
    aligned_r_energy = r_energy.shift(-shift_idx).ffill().bfill()
    aligned_r_pos = shift_array(r_pos, shift_idx)

    return aligned_r_radius, aligned_r_energy, aligned_r_pos


def compute_metrics(h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_energy, aligned_r_energy, k):
    # Radius
    scaled_h_radius = h_radius * k
    rad_error = np.abs(aligned_r_radius - scaled_h_radius)

    # Position
    h_start_pos = h_pos[0]
    r_start_pos = aligned_r_pos[0]
    h_delta_vec = h_pos - h_start_pos
    r_delta_vec = aligned_r_pos - r_start_pos

    # Magnitudes (Scalar displacement)
    h_move_mag = np.linalg.norm(h_delta_vec, axis=1)
    r_move_mag = np.linalg.norm(r_delta_vec, axis=1)

    # FIX: Error is difference of magnitudes, not magnitude of vector difference
    pos_error_var = np.abs(h_move_mag - r_move_mag)

    # Energy
    exp_r_energy = h_energy
    energy_delta = np.abs(aligned_r_energy - exp_r_energy)

    # Energy %
    epsilon = 1e-9
    energy_pct = (energy_delta / (exp_r_energy + epsilon)) * 100.0
    max_energy = exp_r_energy.max()
    mask_steady = exp_r_energy > (max_energy * 0.05)
    energy_pct[~mask_steady] = 0
    energy_pct_smooth = energy_pct.rolling(window=10, center=True).mean().fillna(0)

    return (scaled_h_radius, aligned_r_radius, rad_error), \
        (h_move_mag, r_move_mag, pos_error_var), \
        (exp_r_energy, aligned_r_energy, energy_delta), \
        energy_pct_smooth


class ResultVisualizer:
    def __init__(self, time, radius_data, pos_data, energy_data, en_pct, _fps):
        self.time = time
        self.en_pct = en_pct
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
        self.fig, self.axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True)
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

        # Setup 4th plot (Energy %) - Bottom Right
        ax_last = self.axes_flat[3]
        ax_last.set_title('Relative Energy Error', loc='left', fontweight='bold', fontsize=32)
        ax_last.set_ylabel('Error (%)', fontweight='bold', fontsize=28)
        ax_last.set_xlabel('Time (s)', fontweight='bold', fontsize=28)
        ax_last.tick_params(axis='both', which='major', labelsize=24, length=6, width=2)
        ax_last.set_xlim(time.min(), time.max())
        ax_last.set_ylim(-5, 105)
        ax_last.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
        self.line_pct, = ax_last.plot([], [], color=self.colors['pct'], linewidth=self.lw, label='Rel. Error %')
        ax_last.legend(loc='best', fontsize=24, framealpha=0.95, borderpad=0.8)

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

        self.line_pct.set_data(t_slice, self.en_pct[:idx])
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


def main(csv_path, anim_fps, verbose, no_shift):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    try:
        df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, k = read_csv(csv_path)
    except Exception as e:
        print(f"CSV Reading Error: {e}")
        return

    enable_shift = not no_shift
    aligned_r_radius, aligned_r_energy, aligned_r_pos = preprocess(
        h_radius, r_radius, r_energy, r_pos, verbose, enable_shift
    )

    radius_data, pos_data, energy_data, en_pct = compute_metrics(
        h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_energy, aligned_r_energy, k
    )

    if verbose:
        print("=" * 40 + "\n      GLOBAL RMSE SUMMARY       \n" + "=" * 40)
        print(f"Shifting Enabled:  {enable_shift}")
        print(f"Radius RMSE:       {np.sqrt(np.mean(radius_data[2] ** 2)):.5f} m")
        print(f"Pos Variation RMSE:{np.sqrt(np.mean(pos_data[2] ** 2)):.5f} m")
        print(f"Energy AvgErr:     {np.mean(en_pct):.2f} %")
        print("=" * 40)

    viz = ResultVisualizer(time, radius_data, pos_data, energy_data, en_pct, anim_fps)

    aniext = "_no-shift_video.mp4" if no_shift else "_video.mp4"
    save_path_anim = os.path.splitext(csv_path)[0] + aniext
    viz.save_animation(save_path_anim)

    plotext = "_no-shift.png" if no_shift else ".png"
    save_path_static = os.path.splitext(csv_path)[0] + plotext
    viz.save_static(save_path_static)

    viz.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV file path")
    parser.add_argument("--fps", help="Animation FPS", default=60)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print details")
    parser.add_argument("--no-shift", action="store_true", help="Disable automatic time shifting/alignment")

    args = parser.parse_args()

    main(args.path, args.fps, args.verbose, args.no_shift)