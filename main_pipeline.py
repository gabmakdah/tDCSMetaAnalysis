import subprocess
import sys
import os

def run_script(script_name, args=None):
    """Runs a Python script and handles errors."""
    print(f"\nRunning {script_name}...")
    try:
        command = [sys.executable, script_name]
        if args:
            command.extend(args)
        result = subprocess.run(command, check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    basePath = r'D:\tDCS_PEC_Python'
    base_dir = os.path.join(basePath, 'Code') 
    headmeshes_dir = os.path.join(basePath, 'HeadMeshes')  
    data_filepath = os.path.join(basePath, 'data', 'allData.csv')  
    erniePath = os.path.join(basePath, 'HeadMeshes', 'm2m_ernie')

    scripts = [
        os.path.join(base_dir, "TransformEEGelectrodes.py"),
        os.path.join(base_dir, "simFromCSV_step1.py"),
        os.path.join(base_dir, "meshToNpy_step2.py"),
        # os.path.join(base_dir, "Glasser_Corr_Percentiles_GenMesh_345.py"),
        # os.path.join(base_dir, "runCorrPerm_3.py"),
        os.path.join(base_dir, "Do_Corr_Percentiles_GenMesh_345.py"),
        os.path.join(base_dir, "computePercentiles_4.py"),
        os.path.join(base_dir, "generateMesh_5.py"),
    ]

    print("Starting the SimNIBS pipeline execution.\n")
    folders = os.listdir(headmeshes_dir)
    # torunFolders = folders[8:9]
    torunFolders = [folders[9]]

    # Iterate through all subfolders in HeadMeshes
    for folder in torunFolders:
        subpath = os.path.join(headmeshes_dir, folder)
        print(subpath)
        if os.path.isdir(subpath):
            print(f"Processing folder: {subpath}")
            # eeg_cap = os.path.join(subpath, "EEG10-20_Extended_SPM12.csv")
            # if subpath != erniePath:
            #     run_script(scripts[0], [subpath, erniePath])

            # Run the scripts sequentially for the current folder
            # run_script(scripts[1], [subpath, eeg_cap, data_filepath])
            # run_script(scripts[2], [subpath])
            # run_script(scripts[1], [subpath, eeg_cap, data_filepath])
            # run_script(scripts[2], [subpath])
            run_script(scripts[3], [subpath, data_filepath])
            # run_script(scripts[4], [subpath])
            # run_script(scripts[5], [subpath])

    print("\nPipeline execution completed successfully.")