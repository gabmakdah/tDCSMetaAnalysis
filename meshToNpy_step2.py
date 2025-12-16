import os
import glob
import numpy as np
import simnibs
import shutil

def create_matrice_totale(mesh_dir, overlay_subfolder=None, verbose=False):
    if verbose:
        if overlay_subfolder:
            print(f"Starting to process mesh files in {mesh_dir} (overlay: {overlay_subfolder})...")
        else:
            print(f"Starting to process base mesh files in {mesh_dir}...")

    fields = []
    mesh_files = []
    
    try:
        type_folders = os.listdir(mesh_dir)
    except Exception as e:
        print(f"Error listing directory {mesh_dir}: {e}")
        return None
        
    if verbose:
        print("Found folders:", type_folders)
    
    for type_folder in type_folders:
        type_path = os.path.join(mesh_dir, type_folder)
        if not os.path.isdir(type_path):
            continue
        
        if overlay_subfolder:
            search_folder = os.path.join(type_path, overlay_subfolder)
            pattern = os.path.join(search_folder, '*.msh')
        else:
            search_folder = type_path
            pattern = os.path.join(search_folder, '*.msh')

        if not os.path.exists(search_folder) or not os.path.isdir(search_folder):
            if verbose:
                print(f"Folder {search_folder} does not exist. Skipping...")
            continue

        found_files = glob.glob(pattern)
        if not found_files:
            if verbose:
                print(f"No .msh files found in {search_folder}. Skipping and deleting...")
            shutil.rmtree(type_path)
            continue
        else:
            if verbose:
                print(f"Found {len(found_files)} .msh files in {search_folder}.")
            mesh_files.extend(found_files)

    if not mesh_files:
        print(f"No mesh files found in {mesh_dir} with overlay_subfolder='{overlay_subfolder}'.")
        return None

    try:
        mesh_files = sorted(
            mesh_files,
            key=lambda x: int(os.path.basename(x).split('_')[-2])
        )
    except (IndexError, ValueError) as e:
        if verbose:
            print(f"Warning: Could not sort mesh files as expected. Sorting alphabetically. Error: {e}")
        mesh_files = sorted(mesh_files)

    for file in mesh_files:
        if verbose:
            print(f"Processing {file}...")
        
        try:
            mesh = simnibs.read_msh(file)
        except Exception as e:
            print(f"Error reading {file}: {e}. Skipping this file.")
            shutil.rmtree(os.path.dirname(file))
            continue

        if overlay_subfolder is None:
            print(overlay_subfolder)
            try:
                gray_matter = mesh.crop_mesh(2)
            except KeyError:
                print(f"Label 2 (gray matter) not found in {file}. Skipping this file.")
                continue

            try:
                field_data = gray_matter.field['magnE'][:]
            except KeyError:
                print(f"'magnE' field not found in {file}. Skipping this file.")
                continue
        else:
            try:
                field_data = mesh.field['E_magn'][:]
            except KeyError:
                print(f"'E_magn' field not found in {file}. Skipping this file.")
                continue

        fields.append(field_data)

    if not fields:
        print("No valid field data extracted from mesh files.")
        return None

    try:
        matrice_totale = np.column_stack(fields)
    except ValueError as e:
        print(f"Error stacking fields: {e}")
        return None

    if verbose:
        print(f"Finished processing {len(mesh_files)} mesh files in {mesh_dir} (overlay: {overlay_subfolder}).")
    return matrice_totale

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transform Mesh Files to NPY matrices")
    parser.add_argument("subpath", help="Path to the subject's mesh directory.")
    args = parser.parse_args()

    base_path = os.path.join(args.subpath, 'allMeshes')
    subfolders = ['ToM', 'Altruism', 'Empathy']

    verbose = True
    if verbose:
        print('Starting processing...')
        print("Subfolders:", subfolders)

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_path, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Subfolder {subfolder_path} does not exist. Skipping...")
            continue

        output_files = {
            'base': os.path.join(subfolder_path, f'{subfolder}_matrice_totale_base.npy'),
            'fsavg_overlays': os.path.join(subfolder_path, f'{subfolder}_matrice_totale_fsavg_overlays.npy'),
            'subject_overlays': os.path.join(subfolder_path, f'{subfolder}_matrice_totale_subject_overlays.npy'),
        }

        for overlay_key, out_file in output_files.items():
            if verbose:
                print(f"\nProcessing subfolder: {subfolder} with overlay type: {overlay_key}")

            if overlay_key == 'base':
                overlay = None  
            else:
                overlay = overlay_key  # either 'fsavg_overlays' or 'subject_overlays'
            
            matrice_totale = create_matrice_totale(subfolder_path, overlay_subfolder=overlay, verbose=verbose)
            if matrice_totale is not None:
                np.save(out_file, matrice_totale)
                if verbose:
                    print(f"Saved matrice_totale ({overlay_key}) to {out_file}")
            else:
                error_message = f"Failed to create matrice_totale for {subfolder} with overlay type '{overlay_key}'."
                print(error_message)
                raise RuntimeError(error_message)
