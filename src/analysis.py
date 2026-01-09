"""
Analysis utilities for SMPL parameters from golf swing videos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from pathlib import Path


# SMPL joint indices
JOINT_NAMES = {
    0: 'L_Hip', 1: 'R_Hip', 2: 'Spine1', 3: 'L_Knee', 4: 'R_Knee',
    5: 'Spine2', 6: 'L_Ankle', 7: 'R_Ankle', 8: 'Spine3', 9: 'L_Foot',
    10: 'R_Foot', 11: 'Neck', 12: 'L_Collar', 13: 'R_Collar', 14: 'Head',
    15: 'L_Shoulder', 16: 'R_Shoulder', 17: 'L_Elbow', 18: 'R_Elbow',
    19: 'L_Wrist', 20: 'R_Wrist', 21: 'L_Hand', 22: 'R_Hand'
}

L_ELBOW = 17
R_ELBOW = 18


def rotation_matrix_to_angle(R: np.ndarray) -> float:
    """Convert a 3x3 rotation matrix to the total rotation angle in degrees.

    For a hinge joint like the elbow, this gives the bend angle.
    """
    rot = Rotation.from_matrix(R)
    axis_angle = rot.as_rotvec()
    angle_rad = np.linalg.norm(axis_angle)
    return np.degrees(angle_rad)


def extract_elbow_angles(params_path: str) -> dict:
    """Extract left and right elbow angles from SMPL parameters.

    Args:
        params_path: Path to the .npz file with SMPL parameters

    Returns:
        Dictionary with 'left', 'right', and 'frames' arrays
    """
    params = np.load(params_path)
    body_pose = params['body_pose']  # (T, 23, 3, 3)

    n_frames = body_pose.shape[0]

    left_angles = np.zeros(n_frames)
    right_angles = np.zeros(n_frames)

    for t in range(n_frames):
        left_angles[t] = rotation_matrix_to_angle(body_pose[t, L_ELBOW])
        right_angles[t] = rotation_matrix_to_angle(body_pose[t, R_ELBOW])

    return {
        'left': left_angles,
        'right': right_angles,
        'frames': np.arange(n_frames)
    }


def smooth_signal(signal: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a signal.

    Args:
        signal: Input signal array
        window: Window size (must be odd, larger = smoother)
        polyorder: Polynomial order for fitting

    Returns:
        Smoothed signal
    """
    # Ensure window is odd and not larger than signal
    window = min(window, len(signal))
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2:
        return signal  # Too short to smooth

    return savgol_filter(signal, window, polyorder)


def plot_elbow_angles(params_path: str, fps: float = 30.0, save_path: str = None,
                      smooth_window: int = 7, impact_time: float = None):
    """Plot left and right elbow angles over time with raw scatter and smoothed curves.

    Args:
        params_path: Path to the .npz file with SMPL parameters
        fps: Frames per second (for time axis)
        save_path: If provided, save the plot to this path
        smooth_window: Window size for Savitzky-Golay smoothing
        impact_time: Time of ball impact in seconds (becomes t=0 on the plot)
    """
    angles = extract_elbow_angles(params_path)

    # Convert frames to time
    time = angles['frames'] / fps

    # Shift time so impact_time becomes t=0
    if impact_time is not None:
        time = time - impact_time

    # Smooth the signals
    left_smooth = smooth_signal(angles['left'], window=smooth_window)
    right_smooth = smooth_signal(angles['right'], window=smooth_window)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot vertical line at impact (t=0) if impact_time was specified
    if impact_time is not None:
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, label='Impact')

    # Plot raw data as scatter (semi-transparent)
    ax.scatter(time, angles['left'], c='blue', alpha=0.3, s=30, label='Left (raw)')
    ax.scatter(time, angles['right'], c='red', alpha=0.3, s=30, label='Right (raw)')

    # Plot smoothed curves on top
    ax.plot(time, left_smooth, 'b-', linewidth=2.5, label='Left (smooth)')
    ax.plot(time, right_smooth, 'r-', linewidth=2.5, label='Right (smooth)')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Bend Angle (degrees)', fontsize=12)
    ax.set_title('Elbow Angles During Golf Swing', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits for elbow angles
    ax.set_ylim(0, 180)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    plt.show()

    # Return both raw and smoothed
    angles['left_smooth'] = left_smooth
    angles['right_smooth'] = right_smooth

    return angles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze SMPL parameters from golf swing")
    parser.add_argument("--params", required=True, help="Path to params.npz file")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--save", help="Path to save plot")
    parser.add_argument("--impact-time", type=float, help="Time of ball impact in seconds (becomes t=0)")

    args = parser.parse_args()

    angles = plot_elbow_angles(args.params, fps=args.fps, save_path=args.save,
                               impact_time=args.impact_time)

    print(f"\nElbow angle statistics:")
    print(f"  Left:  min={angles['left'].min():.1f}°, max={angles['left'].max():.1f}°")
    print(f"  Right: min={angles['right'].min():.1f}°, max={angles['right'].max():.1f}°")
