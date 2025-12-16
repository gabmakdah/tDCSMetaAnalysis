import os
import numpy as np
import matplotlib.pyplot as plt


HEAD_MESHES_FOLDER = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes"
OUTPUT_FOLDER = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\Figures\allPECs_plot"

ANALYSIS_TYPES = ["Altruism", "Empathy", "ToM"]



def process_and_plot_from_npy():
    print("Starting data extraction from .npy files and plotting process...")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    try:
        subject_folders = [f for f in os.listdir(HEAD_MESHES_FOLDER) 
                           if f.startswith('m2m_') and os.path.isdir(os.path.join(HEAD_MESHES_FOLDER, f))]
    except FileNotFoundError:
        print(f"ERROR: The head meshes folder was not found at '{HEAD_MESHES_FOLDER}'")
        return

    if not subject_folders:
        print(f"ERROR: No subject folders starting with 'm2m_' found in '{HEAD_MESHES_FOLDER}'")
        return
        
    print(f"Found {len(subject_folders)} subject folders: {sorted(subject_folders)}")

    for analysis_type in ANALYSIS_TYPES:
        print(f"\n--- Processing Analysis Type: {analysis_type} ---")
        pec_data_by_subject = {}
        for subject_folder in sorted(subject_folders):
            subject_name = subject_folder.replace('m2m_', '')
            print(f"  Processing subject: {subject_name}")

            npy_file_path = os.path.join(
                HEAD_MESHES_FOLDER,
                subject_folder,
                'allMeshes',
                'ResultMesh',
                analysis_type,
                f'{analysis_type}_fsavg_overlays_result_mesh.msh.npy'  # Updated filename
            )

            if not os.path.exists(npy_file_path):
                print(f"    WARNING: NPY file not found, skipping. Path: {npy_file_path}")
                continue

            try:
                pec_values = np.load(npy_file_path)
                pec_values_filtered = pec_values[pec_values > 0]

                if pec_values_filtered.size == 0:
                    print(f"    WARNING: After filtering, no non-zero PEC data was found for {subject_name}. Skipping.")
                    continue
                
                print(f"    Successfully loaded and filtered data. Found {pec_values_filtered.size} non-zero values.")
                pec_data_by_subject[subject_name] = pec_values_filtered

            except Exception as e:
                print(f"    ERROR: Could not read or process .npy file {npy_file_path}. Error: {e}")

        if not pec_data_by_subject:
            print(f"No data was collected for {analysis_type}. Skipping plot generation.")
            continue

        subject_labels = list(pec_data_by_subject.keys())
        data_to_plot = list(pec_data_by_subject.values())

        fig, ax = plt.subplots(figsize=(16, 9))
        
        bplot = ax.boxplot(data_to_plot, patch_artist=True, vert=True, labels=subject_labels) 

        ax.set_title(f'Distribution of PEC Magnitudes for {analysis_type}', fontsize=18, pad=20)
        ax.set_ylabel('-PEC Magnitude on Cortical Surface (V/m)', fontsize=14)
        ax.set_xlabel('Subject', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(data_to_plot)))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        plt.tight_layout() 

        output_filename = os.path.join(OUTPUT_FOLDER, f'{analysis_type}_PEC_boxplot.pdf')
        plt.savefig(output_filename, format='pdf', bbox_inches='tight')
        print(f"  > Plot saved to: {output_filename}")
        
        plt.close(fig)

    print("\nAll tasks completed successfully!")


if __name__ == '__main__':
    process_and_plot_from_npy()