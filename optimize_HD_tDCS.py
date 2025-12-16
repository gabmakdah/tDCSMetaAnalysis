from  simnibs import sim_struct as ss
from simnibs import run_simnibs
import simnibs
import numpy as np
from pathlib import Path
import time
import shutil

HEAD_MESH_PATH = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\ernie.msh"
ROI_MESH_PATH = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP\common_significance_reference_subject_ToM_10.msh"
OUTPUT_DIR_STR = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP\OptimizedTDCS_ToM"
TOTAL_ANODE_CURRENT_MA = 2.0  # Anode current in milli-Amps. Cathodes will split the return.
ELECTRODE_DIAMETER_CM = 1.0
ELECTRODE_RADIUS_MM = (ELECTRODE_DIAMETER_CM / 2) * 10
ELECTRODE_DIMS = [ELECTRODE_RADIUS_MM, ELECTRODE_RADIUS_MM]

MONTAGE_MAP = {
    'AFz': ['Nz', 'AF8', 'FCz', 'AF7'],
    'FCz': ['AFz', 'FC4', 'CPz', 'FC3'],
    'FC1': ['AF3', 'FC2', 'CP1', 'FC5'],
    'FC2': ['AF4', 'FC6', 'CP2', 'FC1'],
    'FC3': ['AF3', 'FCz', 'CP3', 'FT7'],
    'FC4': ['AF4', 'FT8', 'CP4', 'FCz'],
    'FC5': ['F7', 'FC1', 'CP5', 'T7'],
    'FC6': ['F8', 'T8', 'CP6', 'FC2'],
    'Cz':  ['Fz', 'C4', 'Pz', 'C3'],
    'C1':  ['F1', 'C2', 'P1', 'C5'],
    'C2':  ['F2', 'C6', 'P2', 'C1'],
    'C3':  ['F3', 'Cz', 'P3', 'T7'],
    'C4':  ['F4', 'T8', 'P4', 'Cz'],
    'C5':  ['F5', 'C1', 'P5', 'T7'],
    'C6':  ['F6', 'T8', 'P6', 'C2'],
    'CPz': ['FCz', 'CP4', 'POz', 'CP3'],
    'CP1': ['FC1', 'CP2', 'PO3', 'CP5'],
    'CP2': ['FC2', 'CP6', 'PO4', 'CP1'],
    'CP3': ['FC3', 'CPz', 'PO3', 'TP7'],
    'CP4': ['FC4', 'TP8', 'PO4', 'CPz'],
    'CP5': ['FC5', 'CP1', 'PO7', 'TP7'],
    'CP6': ['FC6', 'TP8', 'PO8', 'CP2'],
    'Pz':  ['Cz', 'P4', 'Oz', 'P3'],
    'P1':  ['C1', 'P2', 'O1', 'P5'],
    'P2':  ['C2', 'P6', 'O2', 'P1'],
    'P3':  ['C3', 'Pz', 'O1', 'P7'],
    'P4':  ['C4', 'P8', 'O2', 'Pz'],
    'P5':  ['C5', 'P1', 'PO7', 'P7'],
    'P6':  ['C6', 'P8', 'PO8', 'P2']
}

def run_hd_simulation(anode_pos, cathode_positions, session_name, head_mesh_path, output_dir):
    s = ss.SESSION()
    s.fnamehead = str(head_mesh_path)
    s.pathfem = str(output_dir / session_name)

    tdcs_list = s.add_tdcslist()
    num_cathodes = len(cathode_positions)
    cathode_current_ma = -TOTAL_ANODE_CURRENT_MA / num_cathodes
    tdcs_list.currents = [TOTAL_ANODE_CURRENT_MA * 1e-3] + [cathode_current_ma * 1e-3] * num_cathodes
    anode_elec = tdcs_list.add_electrode()
    anode_elec.channelnr = 1
    anode_elec.centre = anode_pos
    anode_elec.shape = 'ellipse' # 'ellipse' with equal dimensions is a circle
    anode_elec.dimensions = ELECTRODE_DIMS
    anode_elec.thickness = 2
    for i,pos in enumerate(cathode_positions):
        cathode_elec = tdcs_list.add_electrode()
        cathode_elec.channelnr = i + 2
        cathode_elec.centre = pos
        cathode_elec.shape = 'ellipse'
        cathode_elec.dimensions = ELECTRODE_DIMS
        cathode_elec.thickness = 2

    s.solver_options = 'pardiso'
    s.open_in_gmsh = False
    s.open_in_simnibs = False
    result_mesh = run_simnibs(s,cpus = 16)
    scalar_result_path = Path(result_mesh.elmdata[0].file_name)
    return scalar_result_path

def evaluate_simulation_with_interpolation(result_mesh_path, roi_mesh_path):
    if not result_mesh_path.is_file():
        print(f"  Evaluation failed: Result file not found at {result_mesh_path}")
        return 0.0

    result_mesh = simnibs.read_msh(str(result_mesh_path))
    roi_mesh = simnibs.read_msh(str(roi_mesh_path))
    roi_element_centroids = roi_mesh.elements.get_element_centers()
    
    if len(roi_element_centroids) == 0:
        print("  WARNING: ROI mesh has no elements to evaluate!")
        return 0.0
        
    e_field_in_roi = result_mesh.elmdata[0].interpolate_to_points(roi_element_centroids)
    mean_e_field_in_roi = np.mean(e_field_in_roi)
    return mean_e_field_in_roi

def main():
    head_mesh_path = Path(HEAD_MESH_PATH)
    roi_mesh_path = Path(ROI_MESH_PATH)
    OUTPUT_DIR = Path(OUTPUT_DIR_STR)


    anode_positions_to_test = list(MONTAGE_MAP.keys())
    num_simulations = len(anode_positions_to_test)
    
    print("Starting targeted HD-tDCS optimization...")
    print(f"Head Mesh: {head_mesh_path.name}")
    print(f"Target ROI Mesh: {roi_mesh_path.name}")
    print(f"Total simulations to run: {num_simulations}")
    print("-" * 60)

    all_results = []
    start_time = time.time()

    for i, anode_pos in enumerate(anode_positions_to_test):
        cathode_positions = MONTAGE_MAP[anode_pos]
        session_name = f"run_{i+1:02d}_Anode-{anode_pos}"
        print(f"\n({i+1}/{num_simulations}) Simulating Montage:")
        print(f"  Anode:   {anode_pos}")
        print(f"  Cathodes: {cathode_positions}")
        try:
            result_path = run_hd_simulation(
                anode_pos, cathode_positions, session_name, head_mesh_path, OUTPUT_DIR
            )
            score = evaluate_simulation_with_interpolation(result_path, roi_mesh_path)
            print(f"  -> Mean E-field in ROI: {score:.4f} V/m")
            all_results.append({
                'anode': anode_pos, 
                'cathodes': cathode_positions, 
                'score': score
            })
        except Exception as e:
            print(f"  ERROR during simulation for Anode {anode_pos}: {e}")
            all_results.append({
                'anode': anode_pos, 
                'cathodes': cathode_positions, 
                'score': 0.0
            })
    end_time = time.time()
    print("\n" + "="*60)
    print("           OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
    
    if not all_results:
        print("\nNo simulations were completed.")
        return
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best_result = all_results[0]

    print("\n--- Best Montage Found ---")
    print(f"  Anode:   {best_result['anode']}")
    print(f"  Cathodes: {best_result['cathodes']}")
    print(f"  Score (Mean E-field): {best_result['score']:.4f} V/m")
    print("\n--- Full Ranking of All Tested Montages ---")
    for i, res in enumerate(all_results):
        print(f"{i+1}. Anode: {res['anode']:<4} | Score: {res['score']:.4f} V/m")

if __name__ == "__main__":
    main()