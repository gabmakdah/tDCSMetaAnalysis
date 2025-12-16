import subprocess, os, pathlib

mesh_path=r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP\common_significance_reference_subject_Empathy_10.msh"
m2m_dir=r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie"
out_dir=r"C:\Users\Gabma\OneDrive\Dokumente\tDCS_PEC_Python\HeadMeshes\m2m_ernie\CombinedP"
os.makedirs(out_dir, exist_ok=True)
out_base=str(pathlib.Path(out_dir)/pathlib.Path(mesh_path).stem)
subprocess.run(["subject2mni","-i",mesh_path,"-m",m2m_dir,"-o",out_base],check=True)






