# caution: install MEArec separately

import MEArec as mr
import time
# template generation
def generating_templates():
    cell_models_folder = mr.get_default_cell_models_folder()
    tempgen = mr.gen_templates(cell_models_folder=cell_models_folder, verbose=True)
    mr.save_template_generator(tempgen=tempgen, filename='demo_templates.h5')

# recording generation
def generating_multiple_recordings(number_of_recordings = 5):
    recording_params = mr.get_default_recordings_params()
    recording_params['templates']['min_dist'] = 5  # 5 in µm
    recording_params['recordings']['fs'] = 12500  # in Hz
    i = 0
    while i < number_of_recordings:
        i = i + 1
        current_timestamp = int(time.time())
        recording_params['seeds']['spiketrains'] = current_timestamp
        recording_params['seeds']['templates'] = current_timestamp
        recording_params['seeds']['convolution'] = current_timestamp
        recording_params['seeds']['noise'] = current_timestamp
        recgen = mr.gen_recordings(params=recording_params, templates=f'demo_templates.h5')
        mr.save_recording_generator(recgen=recgen, filename=f'training_data/demo_recording_{current_timestamp}.h5')
        time.sleep(1)

def generating_recording():
    current_timestamp = int(time.time())
    recording_params = mr.get_default_recordings_params()
    recording_params['templates']['min_dist'] = 5  # 5 in µm
    recording_params['recordings']['fs'] = 12500  # in Hz
    recgen = mr.gen_recordings(params=recording_params, templates=f'demo_templates.h5')
    mr.save_recording_generator(recgen=recgen, filename=f'demo_recording_{current_timestamp}.h5')

generating_templates()
generating_multiple_recordings()
generating_recording()