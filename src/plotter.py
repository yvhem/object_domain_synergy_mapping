import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


# --- Data Processing Functions ---

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Radius, Position and Rotation
    h_radius = df['HumanRadius']
    r_radius = df['RobotRadius']
    h_pos = df[['hPos_x', 'hPos_y', 'hPos_z']].values
    r_pos = df[['rPos_x', 'rPos_y', 'rPos_z']].values
    h_rot = df[['hRot_x', 'hRot_y', 'hRot_z']].values
    r_rot = df[['rRot_x', 'rRot_y', 'rRot_z']].values
    # Energy and Scaling Factor
    h_energy = df['HumanEnergy']
    r_energy = df['RobotEnergy']
    k = df['ScalingFactor'].iloc[0]
    time = df['Time']
    return df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, h_rot, r_rot, k


def preprocess(h_radius, r_radius, r_energy, r_pos, r_rot, verbose):
    h_mid = (h_radius.max() + h_radius.min()) / 2.0
    r_mid = (r_radius.max() + r_radius.min()) / 2.0
    idx_mid_h = (np.abs(h_radius - h_mid)).argmin()
    idx_mid_r = (np.abs(r_radius - r_mid)).argmin()
    shift_idx = idx_mid_r - idx_mid_h

    if verbose:
        print(f"Shift Applied: {shift_idx} frames")

    def shift_array(arr, shift):
        result = np.roll(arr, -shift, axis=0)
        if shift > 0:
            result[-shift:] = result[-shift - 1]
        elif shift < 0:
            result[:-shift] = result[-shift]
        return result

    aligned_r_radius = r_radius.shift(-shift_idx).ffill().bfill()
    aligned_r_energy = r_energy.shift(-shift_idx).ffill().bfill()
    aligned_r_pos = shift_array(r_pos, shift_idx)
    aligned_r_rot = shift_array(r_rot, shift_idx)

    return aligned_r_radius, aligned_r_energy, aligned_r_pos, aligned_r_rot


def compute_metrics(h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_rot, aligned_r_rot, h_energy, aligned_r_energy,
                    k):
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
    var_diff_vec = h_delta_vec - r_delta_vec
    pos_error_var = np.linalg.norm(var_diff_vec, axis=1)
    # Rotation
    rot_h_obj = R.from_euler('xyz', h_rot, degrees=True)
    rot_r_obj = R.from_euler('xyz', aligned_r_rot, degrees=True)
    h_start_rot = rot_h_obj[0]
    r_start_rot = rot_r_obj[0]
    h_rel = rot_h_obj * h_start_rot.inv()
    r_rel = rot_r_obj * r_start_rot.inv()
    h_excursion = np.degrees(h_rel.magnitude())
    r_excursion = np.degrees(r_rel.magnitude())
    diff_rel = r_rel * h_rel.inv()
    rot_error_var = np.degrees(diff_rel.magnitude())
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
        (h_excursion, r_excursion, rot_error_var), \
        (exp_r_energy, aligned_r_energy, energy_delta), \
        energy_pct_smooth


class ResultVisualizer:
    def __init__(self, time, radius_data, pos_data, rot_data, energy_data, en_pct, _fps):
        self.time = time
        self.en_pct = en_pct
        self.fps = _fps

        # Styles
        sns.set_context("paper", font_scale=1.2)
        sns.set_style("whitegrid")
        self.colors = {'h': '#1f77b4', 'r': '#ff7f0e', 'err_f': '#ff9999', 'err_l': '#d62728', 'pct': '#800080'}
        self.lw = 2.5

        # Initialize Figure
        self.fig, self.axes = plt.subplots(5, 1, figsize=(10, 16), sharex=True)

        # Data grouping for iteration
        self.plot_defs = [
            (self.axes[0], "Radius Comparison", "Radius (m)", radius_data),
            (self.axes[1], "Position Variation", "Displacement (m)", pos_data),
            (self.axes[2], "Rotation Variation ", "Excursion (deg)", rot_data),
            (self.axes[3], "Elastic Energy", "Energy (J)", energy_data)
        ]

        self.lines = []

        # Setup first 4 plots
        for ax, title, ylabel, data in self.plot_defs:
            self._setup_axis(ax, title, ylabel, data[0], data[1], data[2])

        # Setup 5th plot
        ax5 = self.axes[4]
        ax5.set_title('Relative Energy Error', loc='left', fontweight='bold')
        ax5.set_ylabel('Error (%)', fontweight='bold')
        ax5.set_xlabel('Time (s)', fontweight='bold')
        ax5.set_xlim(time.min(), time.max())
        ax5.set_ylim(-5, 105)
        ax5.grid(True, linestyle='--', alpha=0.5)
        self.line_pct, = ax5.plot([], [], color=self.colors['pct'], linewidth=2, label='Rel. Error %')
        ax5.legend(loc='upper right')

        plt.tight_layout()

    def _setup_axis(self, ax, title, ylabel, h_data, r_data, e_data):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_xlim(self.time.min(), self.time.max())

        # Calculate Y limits
        all_vals = np.concatenate([h_data, r_data, e_data])
        y_min, y_max = all_vals.min(), all_vals.max()
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Initialize Empty Lines
        l_err, = ax.plot([], [], color=self.colors['err_l'], linewidth=1, alpha=0.6)
        l_h, = ax.plot([], [], color=self.colors['h'], linestyle='--', linewidth=self.lw, label='Human')
        l_r, = ax.plot([], [], color=self.colors['r'], linewidth=self.lw, alpha=0.9, label='Robot')

        ax.legend(loc='upper right', framealpha=0.95)

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

        # Update first 4 subplots
        for item in self.lines:
            item['l_h'].set_data(t_slice, item['d_h'][:idx])
            item['l_r'].set_data(t_slice, item['d_r'][:idx])
            item['l_err'].set_data(t_slice, item['d_err'][:idx])
            for collection in item['ax'].collections:
                collection.remove()
            item['ax'].fill_between(t_slice, 0, item['d_err'][:idx], color=self.colors['err_f'], alpha=0.4)

        # Update 5th subplot
        self.line_pct.set_data(t_slice, self.en_pct[:idx])
        return []

    def save_static(self, path):
        self.update_plot(-1)  # Render full data
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
                # Note: Pillow writer might not support progress_callback in all versions
                ani.save(gif_path, writer='pillow', fps=60, dpi=100)
                print(f"GIF saved to: {gif_path}")
            except Exception as e2:
                print(f"Failed to save animation: {e2}")

    def show(self):
        plt.show()


def main(csv_path, anim_fps, verbose):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    try:
        df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, h_rot, r_rot, k = read_csv(csv_path)
    except Exception as e:
        print(f"CSV Reading Error: {e}")
        return

    aligned_r_radius, aligned_r_energy, aligned_r_pos, aligned_r_rot = preprocess(h_radius, r_radius, r_energy, r_pos, r_rot, verbose)

    # Compute metrics
    radius_data, pos_data, rot_data, energy_data, en_pct = compute_metrics(h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_rot, aligned_r_rot, h_energy, aligned_r_energy, k)

    if verbose:
        print("=" * 40 + "\n      GLOBAL RMSE SUMMARY       \n" + "=" * 40)
        print(f"Radius RMSE:       {np.sqrt(np.mean(radius_data[2] ** 2)):.5f} m")
        print(f"Pos Variation RMSE:{np.sqrt(np.mean(pos_data[2] ** 2)):.5f} m")
        print(f"Rot Variation RMSE:{np.sqrt(np.mean(rot_data[2] ** 2)):.5f} deg")
        print(f"Energy AvgErr:     {np.mean(en_pct):.2f} %")
        print("=" * 40)

    # Initialize Visualizer
    viz = ResultVisualizer(time, radius_data, pos_data, rot_data, energy_data, en_pct, anim_fps)

    # Save Animation
    save_path_anim = os.path.splitext(csv_path)[0] + "_video.mp4"
    viz.save_animation(save_path_anim)

    # Save Static Plot
    save_path_static = os.path.splitext(csv_path)[0] + ".png"
    viz.save_static(save_path_static)

    # Show
    viz.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV file path")
    parser.add_argument("anim_fps", help="Animation FPS", default=60)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print details")
    args = parser.parse_args()

    main(args.path, args.anim_fps, args.verbose)