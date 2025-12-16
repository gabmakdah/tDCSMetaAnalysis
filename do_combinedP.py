import os
import numpy as np
import simnibs
import pyvista as pv
from scipy.spatial import cKDTree

def find_common_significant_nodes(
    m2m_folders,
    mesh_paths,
    field_name='-negLog10Pvalues',
    threshold=1.2,
    reference_index=0,
    transformation_type='nonl'
):
    ref_mesh_path = mesh_paths[reference_index]
    ref_mesh = simnibs.read_msh(ref_mesh_path)
    ref_coords_subj = ref_mesh.nodes[:, :3]
    ref_coords_mni = simnibs.subject2mni_coords(
        ref_coords_subj,
        m2m_folders[reference_index],
        transformation_type=transformation_type
    )
    ref_field_data = ref_mesh.field[field_name][:]
    if ref_field_data is None:
        raise ValueError(f"Field {field_name} not found in reference mesh.")
    ref_significant_mask = (ref_field_data > threshold)
    ref_kdtree = cKDTree(ref_coords_mni)
    for i, m2m_folder in enumerate(m2m_folders):
        if i == reference_index:
            continue
        mesh = simnibs.read_msh(mesh_paths[i])
        coords_subj = mesh.nodes[:, :3]
        coords_mni = simnibs.subject2mni_coords(
            coords_subj,
            m2m_folder,
            transformation_type=transformation_type
        )
        subj_field_data = mesh.field[field_name][:]
        if subj_field_data is None:
            raise ValueError(f"Field {field_name} not found in mesh {mesh_paths[i]}.")
        subj_significant_mask = (subj_field_data > threshold)
        subj_kdtree = cKDTree(coords_mni)
        idx_ref_sign = np.where(ref_significant_mask)[0]
        ref_sign_coords = ref_coords_mni[idx_ref_sign]
        _, idx_nearest = subj_kdtree.query(ref_sign_coords)
        ref_significant_mask[idx_ref_sign] = subj_significant_mask[idx_nearest]
        common_mni_coords = ref_coords_mni[ref_significant_mask]
        
    return ref_mesh, ref_significant_mask, common_mni_coords

def save_mni_nodes_as_mesh(mni_coords, output_path):
    if mni_coords.shape[0] == 0:
        print("Warning: No common significant nodes found. MNI file will not be saved.")
        return
    cloud = pv.PolyData(mni_coords)
    cloud.save(output_path, binary=True)
    print("Wrote common significant MNI nodes to:", output_path)


def map_common_significant_to_reference_subject_space(
    ref_mesh,
    ref_significant_mask,
    m2m_folder_ref,
    output_msh_path,
    transformation_type='nonl'
):
    mask_data = simnibs.NodeData(ref_significant_mask.astype(float), name='common_significance')
    ref_mesh.add_node_field(mask_data, '-common_significance')

    ref_mesh.write(output_msh_path)

if __name__ == "__main__":
    basepath = r"D:\tDCS_PEC_Python\HeadMeshes"
    m2m_folders = [
        os.path.join(basepath, d)
        for d in os.listdir(basepath)
        if d.startswith("m2m_") and os.path.isdir(os.path.join(basepath, d))
    ]
    mesh_paths = [
        os.path.join(folder, "allMeshes", "ResultMesh", "ToM", "ToM_result_mesh.msh")
        for folder in m2m_folders
    ]
    reference_index = 8
    ref_mesh, ref_significant_mask, common_mni_coords = find_common_significant_nodes(
        m2m_folders,
        mesh_paths,
        reference_index=reference_index,
    )
    output_mesh_path = os.path.join(
        m2m_folders[reference_index],
        "common_significance_reference_subject.msh"
    )
    map_common_significant_to_reference_subject_space(
        ref_mesh,
        ref_significant_mask,
        m2m_folders[reference_index],
        output_mesh_path,
        transformation_type='nonl'
    )
    output_mni_path = os.path.join(
        m2m_folders[reference_index],
        "common_significance_MNI_nodes.vtk"
    )
    save_mni_nodes_as_mesh(common_mni_coords, output_mni_path)

    print("Wrote combined significance mask to:", output_mesh_path)
    print("Wrote MNI significance mask to:", output_mni_path)
