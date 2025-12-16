import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

def load_coordinates(file_path):
    data = pd.read_csv(file_path, header=None)
    return data.iloc[:, 1:4].to_numpy(), data

def compute_rigid_transformation(A, B):
    assert A.shape == B.shape, "Input point sets must have the same shape"
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    H = np.dot(A_centered.T, B_centered)
    U, S, Vt = np.linalg.svd(H)
    R_opt = np.dot(Vt.T, U.T)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = np.dot(Vt.T, U.T)
    t_opt = centroid_B - np.dot(R_opt, centroid_A)
    T = np.eye(4)
    T[:3, :3] = R_opt
    T[:3, 3] = t_opt
    return T

def apply_transformation(file_path, transformation_matrix, output_file):
    coords, full_data = load_coordinates(file_path)
    num_points = coords.shape[0]
    coords_homogeneous = np.hstack((coords, np.ones((num_points, 1))))
    transformed_coords = np.dot(transformation_matrix, coords_homogeneous.T).T
    full_data.iloc[:, 1:4] = transformed_coords[:, :3]
    full_data.to_csv(output_file, index=False, header=False)
    print(f"Transformed coordinates saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SimNIBS simulations from CSV input.")
    parser.add_argument("subpath", help="Path to the subject's mesh directory.")
    parser.add_argument("erniePath", help="Path to the Ernie folder.")
    args = parser.parse_args()
    source_file = os.path.join(args.erniePath,'EEG10-10_UI_Jurak_2007.csv')
    target_file = os.path.join(args.subpath, "EEG10-10_UI_Jurak_2007.csv")
    file_to_transform = os.path.join(args.erniePath, "EEG10-20_extended_SPM12.csv")
    output_file = os.path.join(args.subpath, "EEG10-20_Extended_SPM12.csv")
    ernie_coords, _ = load_coordinates(source_file)
    george_coords, _ = load_coordinates(target_file)
    transformation_matrix = compute_rigid_transformation(ernie_coords, george_coords)
    print("Transformation Matrix (4x4):")
    print(transformation_matrix)
    apply_transformation(file_to_transform, transformation_matrix, output_file)
