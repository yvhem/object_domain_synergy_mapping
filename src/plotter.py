import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import argparse


def preprocess(h_radius, r_radius, h_energy, r_energy, k, verbose):
    # Alignment (Half-Max Method)
    h_mid = (h_radius.max() + h_radius.min()) / 2.0
    r_mid = (r_radius.max() + r_radius.min()) / 2.0

    idx_mid_h = (np.abs(h_radius - h_mid)).argmin()
    idx_mid_r = (np.abs(r_radius - r_mid)).argmin()

    shift_idx = idx_mid_r - idx_mid_h

    if verbose:
        print("--- Alignment ---")
        print(f"Shift Applied: {shift_idx} frames")

    # Shift Robot Data
    aligned_r_radius = r_radius.shift(-shift_idx)
    aligned_r_energy = r_energy.shift(-shift_idx)

    # Fill missing values at edges
    aligned_r_radius = aligned_r_radius.ffill().bfill()
    aligned_r_energy = aligned_r_energy.ffill().bfill()

    # Expected robot energy
    exp_r_energy = h_energy * (k ** 2)

    # Compute error
    epsilon = 1e-9
    raw_error = np.abs(aligned_r_energy - exp_r_energy) / (exp_r_energy + epsilon) * 100.0

    # Filter out startup phase (<5% max)
    max_energy = exp_r_energy.max()
    mask_steady = exp_r_energy > (max_energy * 0.05)

    aligned_error_p = raw_error.copy()
    aligned_error_p[~mask_steady] = 0

    # Rolling smoothing
    smoothed_error = aligned_error_p.rolling(window=10, center=True).mean().fillna(0)

    return aligned_r_radius, aligned_r_energy, exp_r_energy, aligned_error_p, smoothed_error


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    h_radius = df['HumanRadius']
    r_radius = df['RobotRadius']
    h_energy = df['HumanEnergy']
    r_energy = df['RobotEnergy']
    k = df['ScalingFactor']
    time = df['Time']
    return df, h_radius, r_radius, h_energy, r_energy, k, time


def plot(time, h_radius, aligned_r_radius, aligned_r_energy, exp_r_energy, aligned_error_p, smoothed_error):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    # Plot 1: Radius
    color_robot = '#C00000'
    color_human = '#0072B2'
    ln1 = ax1.plot(time, aligned_r_radius, label='Robot Radius (Aligned)', color=color_robot, linewidth=2.5, zorder=2)
    ax1.set_ylabel('Robot Radius (m)', fontweight='bold')
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax1_twin = ax1.twinx()
    ln2 = ax1_twin.plot(time, h_radius, label='Human Radius', color=color_human, linestyle='--', linewidth=2, zorder=2)
    ax1_twin.tick_params(axis='y')
    ax1_twin.grid(False)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    leg1 = ax1.legend(lns, labs, loc='center right', frameon=True, framealpha=1.0, edgecolor='black', fancybox=False)
    leg1.set_zorder(10)
    ax1.set_title('A. Hand Closure Comparison', loc='left', fontweight='bold')

    # Plot 2: Energy
    ax2.plot(time, exp_r_energy, label='Expected Energy', color='black', linestyle='--', linewidth=2, zorder=2)
    ax2.plot(time, aligned_r_energy, label='Measured Robot Energy', color='#E69F00', linewidth=2.5, zorder=2)
    ax2.set_ylabel('Elastic Energy (J)', fontweight='bold')
    ax2.set_title('B. Elastic Energy Validation', loc='left', fontweight='bold')
    leg2 = ax2.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='black', fancybox=False)
    leg2.set_zorder(10)
    ax2.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Plot 3: Error
    ax3.plot(time, aligned_error_p, label='Raw Error', color='gray', alpha=0.3, linewidth=1, zorder=1)
    ax3.plot(time, smoothed_error, label='Avg. Error (Smoothed)', color='#800080', linewidth=2.5, zorder=2)
    ax3.axhline(y=25, color='black', linestyle=':', linewidth=1.5, label='Steady State (~25%)', zorder=1)
    if smoothed_error.max() > 100: ax3.set_ylim(-5, 110)
    ax3.set_ylabel('Error (%)', fontweight='bold')
    ax3.set_xlabel('Time (s)', fontweight='bold')
    ax3.set_title('C. Relative Energy Error', loc='left', fontweight='bold')
    leg3 = ax3.legend(loc='upper right', frameon=True, framealpha=1.0, edgecolor='black', fancybox=False)
    leg3.set_zorder(10)
    ax3.grid(True, linestyle='--', alpha=0.5, zorder=0)

    plt.tight_layout()
    return plt


def main(csv_path, verbose):
    if not os.path.exists(csv_path):
        print(f"Error: The file {csv_path} was not found.")
        return

    save_path = os.path.splitext(csv_path)[0] + ".png"

    df, h_radius, r_radius, h_energy, r_energy, k, time = read_csv(csv_path)

    aligned_r_radius, aligned_r_energy, exp_r_energy, aligned_error_p, smoothed_error = preprocess(
        h_radius, r_radius, h_energy, r_energy, k, verbose
    )

    plt_obj = plot(time, h_radius, aligned_r_radius, aligned_r_energy,
                   exp_r_energy, aligned_error_p, smoothed_error)

    plt_obj.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Chart saved to:", save_path)
    plt_obj.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="CSV file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed info")
    args = parser.parse_args()

    main(args.path, args.verbose)