import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def smooth_signal(data, window_size):
    """
    Smooths a 1D or 2D numpy array using a centered rolling average.
    Uses pandas for robust edge handling (min_periods=1).
    """
    if window_size <= 1:
        return data

    if data.ndim == 1:
        return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().values
    elif data.ndim == 2:
        return pd.DataFrame(data).rolling(window=window_size, min_periods=1, center=True).mean().values

    return data


def read_csv(csv_path, smooth_win=15):
    df = pd.read_csv(csv_path)

    # Radius
    h_radius = df['HumanRadius'].values * 100.0  # Convert M -> CM
    r_radius = df['RobotRadius'].values * 100.0  # Convert M -> CM

    # Position
    h_pos = df[['hPos_x', 'hPos_y', 'hPos_z']].values * 100.0  # Convert M -> CM
    r_pos = df[['rPos_x', 'rPos_y', 'rPos_z']].values * 100.0  # Convert M -> CM

    # Energy
    h_energy = df['HumanEnergy'].values
    r_energy = df['RobotEnergy'].values
    k = df['ScalingFactor'].iloc[0]
    time = df['Time']

    if smooth_win > 1:
        print(f"Smoothing data with window size: {smooth_win}")
        h_radius = smooth_signal(h_radius, smooth_win)
        r_radius = smooth_signal(r_radius, smooth_win)
        h_pos = smooth_signal(h_pos, smooth_win)
        r_pos = smooth_signal(r_pos, smooth_win)
        h_energy = smooth_signal(h_energy, smooth_win)
        r_energy = smooth_signal(r_energy, smooth_win)

    return df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, k


def compute_metrics(h_radius, aligned_r_radius, h_pos, aligned_r_pos, h_energy, aligned_r_energy, k):
    # Radius
    scaled_h_radius = h_radius * k
    rad_error = np.abs(aligned_r_radius - scaled_h_radius)

    # Position
    h_start_pos = h_pos[0]
    r_start_pos = aligned_r_pos[0]
    h_disp_vec = h_pos - h_start_pos
    r_disp_vec = aligned_r_pos - r_start_pos
    target_r_disp_vec = h_disp_vec * k
    error_vec = target_r_disp_vec - r_disp_vec

    traj_error = np.linalg.norm(error_vec, axis=1)
    h_move_mag = np.linalg.norm(target_r_disp_vec, axis=1)
    r_move_mag = np.linalg.norm(r_disp_vec, axis=1)

    # Energy
    exp_r_energy = h_energy
    energy_delta = np.abs(aligned_r_energy - exp_r_energy)

    return (scaled_h_radius, aligned_r_radius, rad_error), \
        (h_move_mag, r_move_mag, traj_error), \
        (exp_r_energy, aligned_r_energy, energy_delta)


class ResultVisualizer:
    def __init__(self, time, radius_data, pos_data, energy_data, _fps):
        self.time = time
        self.fps = _fps

        # Styles
        sns.set_context("paper", font_scale=2.5)
        plt.rcParams.update({
            'font.size': 35,
            'axes.titlesize': 35,
            'axes.labelsize': 26,
            'xtick.labelsize': 35,
            'ytick.labelsize': 35,
            'legend.fontsize': 25,
            'lines.linewidth': 4
        })
        sns.set_style("whitegrid")

        self.colors = {'h': '#1f77b4', 'r': '#ff7f0e', 'err_f': '#ff9999', 'err_l': '#d62728'}

        # Layout: 2x2 grid
        self.fig = plt.figure(figsize=(23, 14))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1], sharex=ax1)
        ax3 = self.fig.add_subplot(gs[1, :], sharex=ax1)

        self.axes_flat = [ax1, ax2, ax3]

        self.plot_defs = [
            (self.axes_flat[0], "Radius Variation", "Radius (cm)", radius_data, "Human", "Robot"),
            (self.axes_flat[1], "Position Drift", "Displacement (cm)", pos_data, "Human", "Robot"),
            (self.axes_flat[2], "Elastic Energy", "Energy (J)", energy_data, "Human", "Robot")
        ]

        self.lines = []

        for ax, title, ylabel, data, lbl_h, lbl_r in self.plot_defs:
            self._setup_axis(ax, title, ylabel, data[0], data[1], data[2], lbl_h, lbl_r)

        plt.tight_layout()

    def _setup_axis(self, ax, title, ylabel, h_data, r_data, e_data, lbl_h, lbl_r):
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlim(self.time.min(), self.time.max())
        ax.set_xlabel('Time (s)', fontweight='bold')

        l_h, = ax.plot(self.time, h_data, color=self.colors['h'], linestyle='--', label=lbl_h)
        l_r, = ax.plot(self.time, r_data, color=self.colors['r'], alpha=0.9, label=lbl_r)

        ax.fill_between(self.time, 0, e_data, color=self.colors['err_f'], alpha=0.5, label='Error', linewidth=0)

        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, linestyle='--', alpha=0.5)

        self.lines.append({'ax': ax, 'l_h': l_h, 'l_r': l_r,
                           'd_h': h_data, 'd_r': r_data, 'd_err': e_data})

    def update_plot(self, frame_idx):
        idx = len(self.time) if frame_idx == -1 else frame_idx
        t_slice = self.time.iloc[:frame_idx] if frame_idx != -1 else self.time

        for item in self.lines:
            # Update lines
            item['l_h'].set_data(t_slice, item['d_h'][:idx])
            item['l_r'].set_data(t_slice, item['d_r'][:idx])

            # Update Area
            for coll in item['ax'].collections:
                coll.remove()

            item['ax'].fill_between(t_slice, 0, item['d_err'][:idx],
                                    color=self.colors['err_f'], alpha=0.5, linewidth=0)
        return []

    def save_plot(self, path):
        self.update_plot(-1)
        self.fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {path}")

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


def main(path, fps=60, verbose=False, mode=1, smooth=0):
    csv_paths = set()
    if path == 'b':
        csv_paths.add('Test/Barrett/Barrett_GraspValidation.csv')
    elif path == 'm':
        csv_paths.add('Test/Mia/Mia_GraspValidation.csv')
    elif path == 's':
        csv_paths.add('Test/Shadow/Shadow_GraspValidation.csv')
    elif path == 'all':
        csv_paths = ['Test/Barrett/Barrett_GraspValidation.csv', 'Test/Mia/Mia_GraspValidation.csv',
                     'Test/Shadow/Shadow_GraspValidation.csv']
    else:
        csv_paths.add(path)

    for csv_path in tqdm(csv_paths):
        if not os.path.exists(csv_path): return
        try:
            df, time, h_radius, r_radius, h_energy, r_energy, h_pos, r_pos, k = read_csv(csv_path, smooth)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return

        radius_data, pos_data, energy_data = compute_metrics(
            h_radius, r_radius, h_pos, r_pos, h_energy, r_energy, k
        )

        if verbose:
            print(f"Radius RMSE:    {np.sqrt(np.mean(radius_data[2] ** 2)):.2f} cm")
            print(f"Pos Drift RMSE: {np.sqrt(np.mean(pos_data[2] ** 2)):.2f} cm")

        viz = ResultVisualizer(time, radius_data, pos_data, energy_data, fps)

        if mode in (1, 3):
            save_path_plot = os.path.splitext(csv_path)[0] + ".png"
            viz.save_plot(save_path_plot)

        if mode in (2, 3):
            save_path_anim = os.path.splitext(csv_path)[0] + "_video.mp4"
            viz.save_animation(save_path_anim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to CSV file or names: 'b, m, s, all'")
    parser.add_argument("--fps", default=60, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument( "--mode", type=int, choices=[1,2,3], default=1 , help="1. Plots; 2. Animation; 3. Both")
    parser.add_argument("--smooth", type=int, default=0, help="Window size for smoothing (0 to disable)")

    args = parser.parse_args()
    main(args.path, args.fps, args.verbose, args.mode, args.smooth)



