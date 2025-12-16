from simnibs import sim_struct, run_simnibs
import pandas as pd
import os
import concurrent.futures
import sys

def addElectrode(tdcsList, electrodeLocation, electrodeSize, electrodeShape, electrodeThickness, electrodeYdir, electrodeHole, channelType):
    # channelnr : 1 = cathode, 2 = anode
    if ',' in electrodeLocation:
        try:
            electrodeLocation = [float(x) for x in electrodeLocation.split(',')]
            isNotVector = 0
        except:
            isNotVector = 1
        
        if isNotVector == 1:
            for oneLocation in electrodeLocation.split(','):
                tdcsList = addElectrode(tdcsList, oneLocation, electrodeSize, electrodeShape, electrodeThickness, electrodeYdir, electrodeHole, channelType)
    else:
        electrode = tdcsList.add_electrode()
        electrode.channelnr = 1 if channelType == "cathode" else 2
        electrode.dimensions = [int(x) for x in electrodeSize.split('x')]
        electrode.shape = electrodeShape
        electrode.thickness = electrodeThickness
        electrode.centre = electrodeLocation

    if not pd.isna(electrodeHole):
        hole = electrode.add_hole()
        hole.shape = 'ellipse'
        hole.dimensions = [int(x) for x in electrodeHole.split('x')]
        hole.centre = [0, 0]

    return tdcsList

def run_simulation_for_type(attr_type, df, base_path, subpath, eeg_cap):
    filtered_df = df[df['Type'] == attr_type]
    
    output_dir = os.path.join(base_path, attr_type)
    os.makedirs(output_dir, exist_ok=True)
    
    unique_studies = filtered_df['Name'].unique()
    unique_studies_df = pd.DataFrame(unique_studies, columns=['StudyName'])
    unique_studies_df.to_csv(os.path.join(output_dir, 'unique_studies.csv'), index=False)
    print(subpath)
    s = sim_struct.SESSION()

    s.subpath = subpath  
    s.pathfem = output_dir  
    s.eeg_cap = eeg_cap  
    s.open_in_gmsh = False
    lastIndex = 0
    s.interpolate_to_surface = True  
    s.transform_to_fsaverage = True  
    for index, row in filtered_df.iterrows():
        uniqueIndex = row['Name']
        if lastIndex != uniqueIndex:
            tdcslist = s.add_tdcslist()
            tdcslist.currents = [-row['mA'] * 1e-3, row['mA'] * 1e-3]
            tdcslist = addElectrode(tdcslist, row['aLocation'], row['aSize'], row['Shape'], row['aThickness'], row['aY'], row['aHole'], "anode")
            tdcslist = addElectrode(tdcslist, row['cLocation'], row['cSize'], row['Shape'], row['cThickness'], row['cY'], row['cHole'], "cathode")
        lastIndex = uniqueIndex

    run_simnibs(s, n_proc=16)

def run_simulation_for_study(study, base_path, subpath, eeg_cap):
    import simnibs
    from simnibs import sim_struct, run_simnibs
    import os

    output_dir = os.path.join(base_path, study['Type'], study['Name'])
    os.makedirs(output_dir, exist_ok=True)

    s = sim_struct.SESSION()
    s.subpath = subpath
    s.pathfem = output_dir
    s.eeg_cap = eeg_cap
    s.open_in_gmsh = False
    s.map_to_fsavg = True 
    s.map_to_surf = True 
    tdcslist = s.add_tdcslist()
    tdcslist.currents = [-study['mA'] * 1e-3, study['mA'] * 1e-3]
    tdcslist = addElectrode(tdcslist, study['aLocation'], study['aSize'], study['Shape'], study['aThickness'], study['aY'], study['aHole'], "anode")
    tdcslist = addElectrode(tdcslist, study['cLocation'], study['cSize'], study['Shape'], study['cThickness'], study['cY'], study['cHole'], "cathode")

    run_simnibs(s, cpus=16)


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Run SimNIBS simulations from CSV input.")
    parser.add_argument("subpath", help="Path to the subject's mesh directory.")
    parser.add_argument("eeg_cap", help="Path to the EEG cap positions file.")
    parser.add_argument("data_filepath", help="Path to the CSV data file.")
    args = parser.parse_args()

    df = pd.read_csv(args.data_filepath)

    listAttributeTypes = ['ToM', 'Altruism', 'Empathy']
    base_path = os.path.join(args.subpath, 'allMeshes')

    filtered_df = df[df['Type'].isin(listAttributeTypes)]

    studies = filtered_df.to_dict('records')

    max_workers = 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_study = {
            executor.submit(run_simulation_for_study, study, base_path, args.subpath, args.eeg_cap): study
            for study in studies
        }

        for future in as_completed(future_to_study):
            study = future_to_study[future]
            try:
                future.result()
                print(f"Simulation completed for study: {study['Name']}")
            except Exception as exc:
                print(f"Simulation generated an exception for study {study['Name']}: {exc}")