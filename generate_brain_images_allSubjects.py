import pyvista as pv
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def create_asymmetric_colormap(cmap_name, vmin, vcenterpre, vcenter, vmax):
    """Creates an asymmetric colormap, useful for p-values."""
    norm_center = (vcenter - vmin) / (vmax - vmin)
    norm_precenter = (vcenterpre - vmin) / (vmax - vmin)
    original_cmap = plt.cm.get_cmap(cmap_name)
    colors = [original_cmap(0.0), original_cmap(0.0), original_cmap(0.5), original_cmap(1.0)]
    nodes = [0.0, norm_precenter, norm_center, 1.0]
    new_cmap_name = f"asymmetric_{cmap_name}"
    new_cmap = mcolors.LinearSegmentedColormap.from_list(new_cmap_name, list(zip(nodes, colors)))
    return new_cmap

def render_single_view(mesh, variable_name, view_params, cmap, clim):
    """Renders a single view of a mesh with specific settings and returns an image."""
    actual_var_name = next((key for key in mesh.point_data if variable_name.strip() in key), None)
    if actual_var_name is None:
        print(f"Warning: Could not find variable '{variable_name}' in mesh. Plotting blank.")
        return np.full((800, 800, 3), 255, dtype=np.uint8)
    plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
    plotter.set_background('white')
    plotter.add_mesh(mesh, scalars=actual_var_name, cmap=cmap, clim=clim, show_scalar_bar=False)
    getattr(plotter, f"view_{view_params['view']}")(negative=view_params.get('negative', False))
    plotter.camera.zoom(1.3)
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img

def generate_multi_subject_grid(subject_paths, output_dir, plot_settings):
    """
    Generates a single large PDF figure with a grid of plots for all subjects,
    including horizontal colorbars with correctly displayed labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Pre-computation Phase ---")
    loaded_meshes, global_avg_mesh_max, global_pec_min, global_pec_max = {}, 0, float('inf'), float('-inf')
    subject_names = sorted(subject_paths.keys())
    for name in subject_names:
        path = subject_paths[name]
        try:
            mesh = pv.read(path)
            loaded_meshes[name] = mesh
            avg_mesh_key = next((k for k in mesh.point_data if '-averageMesh' in k), None)
            if avg_mesh_key: global_avg_mesh_max = max(global_avg_mesh_max, mesh.point_data[avg_mesh_key].max())
            pec_key = next((k for k in mesh.point_data if '-PEC' in k), None)
            if pec_key:
                global_pec_min = min(global_pec_min, mesh.point_data[pec_key].min())
                global_pec_max = max(global_pec_max, mesh.point_data[pec_key].max())
        except Exception as e: print(f"Could not load or process mesh for {name}: {e}")
    plot_settings['-averageMesh']['clim'] = [0, global_avg_mesh_max]
    plot_settings['-PEC']['clim'] = [global_pec_min, global_pec_max]
    print("\n--- Global Color Limits ---")
    print(f"averageMesh range: {plot_settings['-averageMesh']['clim']}")
    print(f"PEC range:         {plot_settings['-PEC']['clim']}")
    print(f"negLog10Pvalues:   {plot_settings['-negLog10Pvalues']['clim']} (fixed)\n")

    print("--- Plotting Phase ---")
    variables_in_order = ['-averageMesh', '-PEC', '-negLog10Pvalues']
    views = {'Left': {'view': 'yz'}, 'Top': {'view': 'xy'}, 'Right': {'view': 'yz', 'negative': True}}
    view_keys, n_views = list(views.keys()), len(views.keys())
    n_rows, n_cols = len(subject_names), len(variables_in_order) * n_views
    
    height_ratios = [1] * n_rows + [0.15]
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(20, 2 * n_rows + 2), facecolor='white', gridspec_kw={'height_ratios': height_ratios})
    if n_rows == 1: axes = np.array(axes).reshape(2, n_cols)
    
    for row_idx, subject_name in enumerate(subject_names):
        print(f"  Processing row {row_idx + 1}/{n_rows}: {subject_name}")
        mesh = loaded_meshes.get(subject_name)
        if mesh is None:
            for col_idx in range(n_cols): axes[row_idx, col_idx].axis('off')
            continue
        display_name = subject_name.replace("m2m_", "")
        ax_for_label = axes[row_idx, 0]
        ax_for_label.text(-0.1, 0.5, display_name, transform=ax_for_label.transAxes, ha='right', va='center', rotation=90, fontsize=14)
        for var_idx, var_name in enumerate(variables_in_order):
            for view_idx, view_key in enumerate(view_keys):
                col_idx = var_idx * n_views + view_idx
                ax = axes[row_idx, col_idx]
                settings = plot_settings[var_name]
                img = render_single_view(mesh, var_name, views[view_key], cmap=settings['cmap'], clim=settings['clim'])
                ax.imshow(img)
                ax.axis('off')
                if row_idx == 0:
                    clean_var_name = var_name.replace('-', '').replace('averageMesh', 'AvgMesh')
                    title = f"{clean_var_name}\n{view_key}"
                    ax.set_title(title, fontsize=14, pad=20)
    
    print("\nAdding colour-bars to the dedicated bottom row...")
    if n_rows > 0:
        for ax in axes[n_rows, :]:
            ax.axis('off')

        for var_idx, var_name in enumerate(variables_in_order):
            settings       = plot_settings[var_name]
            vmin, vmax     = settings['clim']
            cmap           = settings['cmap']
            norm           = mcolors.Normalize(vmin=vmin, vmax=vmax)
            mappable       = cm.ScalarMappable(norm=norm, cmap=cmap)

            start_col      = var_idx * n_views
            cax            = axes[n_rows, start_col + n_views // 2]  # middle cell
            cbar           = fig.colorbar(mappable, cax=cax, orientation='horizontal')

          
            cbar.ax.set_xticks([])             
            cbar.outline.set_visible(False)     

            cbar.ax.text(
                0.0, -0.45,                    
                f'{vmin:.2f}',
                transform=cbar.ax.transAxes,
                ha='center', va='top', fontsize=10)

            cbar.ax.text(
                1.0, -0.45,
                f'{vmax:.2f}',
                transform=cbar.ax.transAxes,
                ha='center', va='top', fontsize=10)
            clean_name = var_name.replace('-', '').replace('averageMesh', 'AvgMesh')
            cbar.set_label(clean_name, fontsize=12, labelpad=5, weight='bold')
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95], pad=1.0)

    output_filename = os.path.join(output_dir, "All_Subjects_Summary_Grid.pdf")
    print(f"\nSaving combined figure to: {output_filename}")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("Processing complete!")


if __name__ == "__main__":
    head_meshes_dir = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes"
    output_directory = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\AutomatedFigures"
    subject_mesh_paths = {}
    search_pattern = os.path.join(head_meshes_dir, "m2m_*")
    subject_folders = glob.glob(search_pattern)
    for folder in subject_folders:
        if os.path.isdir(folder):
            subject_name = os.path.basename(folder)
            mesh_file = os.path.join(folder, 'allMeshes', 'ResultMesh', 'Altruism', 'Altruism_result_mesh.msh')
            if os.path.exists(mesh_file):
                subject_mesh_paths[subject_name] = mesh_file
            else:
                print(f"Warning: Mesh file not found for subject {subject_name} at expected path.")

    if not subject_mesh_paths:
        print("Error: No subject mesh files were found. Please check 'head_meshes_dir'.")
    else:
        print(f"Found {len(subject_mesh_paths)} subjects to process.")
        neglogp_cmap = create_asymmetric_colormap(cmap_name='coolwarm', vmin=0.0, vcenterpre=1.0, vcenter=1.2, vmax=1.5)
        plot_settings_dict = {
            '-averageMesh': {'cmap': 'viridis'},
            '-PEC': {'cmap': 'jet'},
            '-negLog10Pvalues': {'cmap': neglogp_cmap, 'clim': [0.0, 1.5]}
        }
        generate_multi_subject_grid(subject_mesh_paths, output_directory, plot_settings_dict)