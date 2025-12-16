import pyvista as pv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_4_view_figure(mesh_path, variable_name, output_path):
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at '{mesh_path}'")
        return

    print(f"Processing: {os.path.basename(mesh_path)}")
    try:
        mesh = pv.read(mesh_path)
    except Exception as e:
        print(f"Failed to load mesh file. Error: {e}")
        return

    if variable_name not in mesh.point_data:
        print(f"Error: Variable '{variable_name}' not found in the mesh.")
        print(f"Available variables: {list(mesh.point_data.keys())}")
        return

    views = {
        'Left':  {'view': 'yz', 'negative': True},
        'Front':    {'view': 'xz', 'negative': True},
        'Top':  {'view': 'xy'},
        'Right':   {'view': 'yz'},
    }
    view_order = ['Left', 'Top', 'Front', 'Right']
    blue_red_cmap = mcolors.LinearSegmentedColormap.from_list("BlueRedCmap", ["blue", "red"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='white')

    for i, view_name in enumerate(view_order):
        ax = axes[i]
        
        plotter = pv.Plotter(off_screen=True, window_size=[800, 800])
        plotter.set_background('white')

        mesh_kwargs = {
            'scalars': variable_name,
            'cmap': blue_red_cmap,
            'clim': [0, 1],  
        }
        if view_name == 'Right':
             mesh_kwargs['scalar_bar_args'] = {'title': variable_name.replace('-', ' ').strip()}
        else:
             mesh_kwargs['show_scalar_bar'] = False

        plotter.add_mesh(mesh, **mesh_kwargs)
        camera_params = views[view_name]
        getattr(plotter, f"view_{camera_params['view']}")(negative=camera_params.get('negative', False))
        plotter.camera.zoom(1.4)

        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(view_name, fontsize=16, pad=10)
    fig_title = os.path.basename(mesh_path).replace('.msh', '').replace('_', ' ')
    fig.suptitle(fig_title, fontsize=20, y=1.02)
    
    fig.tight_layout()
    print(f"  -> Saving figure to: {output_path}")
    plt.savefig(output_path, bbox_inches='tight', dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    files_to_process = {
        "ToM": r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP\common_significance_reference_subject_ToM_10.msh",
        "Empathy": r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP\common_significance_reference_subject_Empathy_10.msh"
    }
    
    variable_field = "-common_significance"
    output_directory = r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\AutomatedFigures\Significance"
    os.makedirs(output_directory, exist_ok=True)
    print("--- Starting Figure Generation ---")
    for name, file_path in files_to_process.items():
        output_filename = os.path.join(output_directory, f"{name}_significance_views-NEW.pdf")
        generate_4_view_figure(
            mesh_path=file_path,
            variable_name=variable_field,
            output_path=output_filename
        )
        
    print("\n--- All tasks complete! ---")