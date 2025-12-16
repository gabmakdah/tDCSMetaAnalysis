import pyvista as pv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def create_asymmetric_colormap(cmap_name, vmin, vcenterpre, vcenter, vmax):
    norm_center = (vcenter - vmin) / (vmax - vmin)
    norm_precenter = (vcenterpre - vmin) / (vmax - vmin)
    original_cmap = cm.get_cmap(cmap_name)
    colors = [
        original_cmap(0.0), original_cmap(0.0), 
        original_cmap(0.5), original_cmap(1.0)
    ]
    nodes = [0.0, norm_precenter, norm_center, 1.0]
    new_cmap_name = f"asymmetric_{cmap_name}"
    new_cmap = mcolors.LinearSegmentedColormap.from_list(new_cmap_name, list(zip(nodes, colors)))
    return new_cmap

def generate_summary_figure_pdf(mesh_path, output_dir, variables_in_order, plot_settings_dict):
    """
    Creates a summary PDF figure with larger, text-free color bars and prints
    the color limits to the terminal.
    """
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at '{mesh_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading mesh: {mesh_path}")
    try:
        mesh = pv.read(mesh_path)
    except Exception as e:
        print(f"Failed to load mesh file. Error: {e}")
        return

    print("Dynamically setting color range for -averageMesh...")
    avg_mesh_var_key = next((key for key in mesh.point_data if '-averageMesh' in key), None)
    if avg_mesh_var_key:
        max_val = mesh.point_data[avg_mesh_var_key].max()
        plot_settings_dict['-averageMesh']['clim'] = [0, max_val]
    else:
        print("  -> Warning: Could not find '-averageMesh' data in mesh.")

    n_rows = len(variables_in_order)
    n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10), facecolor='white')

    views = {
        'Top': {'view': 'xy'},
        'Left':    {'view': 'yz'},
        'Right':   {'view': 'yz', 'negative': True}
    }
    view_keys = list(views.keys())

    for row, var_name in enumerate(variables_in_order):
        actual_var_name = next((key for key in mesh.point_data if var_name.strip() in key), None)
        
        if actual_var_name is None:
            print(f"Warning: Could not find variable '{var_name}' in mesh. Skipping row.")
            for col in range(n_cols):
                axes[row, col].axis('off')
            continue
            
        settings = plot_settings_dict.get(var_name, {})
        custom_cmap = settings.get('cmap', 'viridis')
        custom_clim = settings.get('clim', None)

        if custom_clim:
            min_limit, max_limit = custom_clim
        else: 
            data_array = mesh.point_data[actual_var_name]
            min_limit = data_array.min()
            max_limit = data_array.max()
        print(f"  Color limits for '{actual_var_name.strip()}': [{min_limit:.4f}, {max_limit:.4f}]")

        for col, view_name in enumerate(view_keys):            
            plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
            plotter.set_background('white')

            mesh_kwargs = {
                'scalars': actual_var_name, 'cmap': custom_cmap, 'clim': custom_clim,
            }
            
            if col == n_cols - 1:
                mesh_kwargs['scalar_bar_args'] = {
                    'title': '',         
                    'n_labels': 0,      
                    'width': 1,       
                    'height': 0.15,      
                    'position_x': 0,  
                }
                # ------------------------------------------------
            else:
                mesh_kwargs['show_scalar_bar'] = False
            
            plotter.add_mesh(mesh, **mesh_kwargs)
            
            camera_params = views[view_name]
            getattr(plotter, f"view_{camera_params['view']}")(negative=camera_params.get('negative', False))
            plotter.camera.zoom(1.3)

            img = plotter.screenshot(return_img=True)
            plotter.close()

            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')

            if row == 0:
                ax.set_title(view_name, fontsize=16, pad=10)
        
        axes[row, 0].set_ylabel(var_name.replace('-', ''), fontsize=16, rotation=90, labelpad=20)

    fig.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)

    base_filename = os.path.splitext(os.path.basename(mesh_path))[0]
    output_filename = os.path.join(output_dir, f"{base_filename}_summary.pdf")

    print(f"\nSaving figure to: {output_filename}")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("Processing complete!")


if __name__ == "__main__":
    mesh_file = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\allMeshes\ResultMesh\Empathy\Empathy_result_mesh.msh"
    output_directory = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\AutomatedFigures"
    variables_to_plot_ordered = ['-averageMesh', '-PEC', '-negLog10Pvalues']
    neglogp_cmap = create_asymmetric_colormap(
        cmap_name='coolwarm', vmin=0.0, vcenterpre=1.0, vcenter=1.2, vmax=1.5
    )
    plot_settings = {
        '-averageMesh': {'cmap': 'viridis'},
        '-PEC': {'cmap': 'jet'},
        '-negLog10Pvalues': {'cmap': neglogp_cmap, 'clim': [0.0, 1.5]}
    }
    generate_summary_figure_pdf(mesh_file, output_directory, variables_to_plot_ordered, plot_settings)