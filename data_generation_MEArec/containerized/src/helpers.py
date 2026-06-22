import subprocess
import MEArec as mr
import matplotlib.pyplot as plt

def print_installed_packages():
    result = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

def save_recording_plot_to_disk(path_recording_file, path_output_file='recordings_plot.png'):
    recgen = mr.load_recordings(path_recording_file)
    ax = mr.tools.plot_recordings(recgen)# , overlay_templates=True) # start_time=05.010, end_time=05.050, )
    fig = ax.figure
    fig.savefig(path_output_file, dpi=300, bbox_inches='tight')    

