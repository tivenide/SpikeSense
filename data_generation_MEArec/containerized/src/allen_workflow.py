# Here are some basic workflows for generating MEArec recordings of allen models.
# Functions need specific file structures and corresponding cell models of the allen database.
# Link to data portal: https://celltypes.brain-map.org/data

# To download specific models by model ID:
# https://celltypes.brain-map.org/neuronal_model/download/472300877
# The model ID differs from the specimen_id

# To get to a specific cell by specimen_id:
# https://celltypes.brain-map.org/experiment/electrophysiology/320668879


def generate_template_library_set_0001():
    from allen_functions import generating_template_of_allen_models
    models_path = 'allen/models/set_0001'
    path_to_template_library = 'allen/templatelibs/templates_allen_set0001_seed_0_n500.h5'

    probe_name = 'SqMEA-8-200-60'
    seed = 0

    cell_types = {
        '489932682': 'spiny',
        '485720587': 'spiny',
        '489932435': 'spiny',
        '478048947': 'spiny',
        '472300877': 'spiny',
        '488462965': 'aspiny',
        '478513461': 'aspiny',
        '480631187': 'aspiny',
        '472450023': 'aspiny',
        # '472440759': 'aspiny', # pot. replacement for 472450023
        '479234670': 'aspiny'
    }
    generating_template_of_allen_models(models_path=models_path, export_filename=path_to_template_library, probe_name=probe_name, cell_types=cell_types, seed=seed)


def generate_recordings_set_0001():
    from allen_functions import generate_multiple_recordings_with_different_seeds_and_save_to_disk
    path_template = 'allen/templatelibs/templates_allen_set0001_seed_0_n500.h5'
    path_target = 'allen/recordings/set_0001'

    number_of_recordings_each_parameter_stack = 64
    duration = 60
    fs = 10000
    n = 10
    min_amp = 20
    max_amp = 32
    n_inh = 12
    n_exc = 17
    seed = 10000
    generate_multiple_recordings_with_different_seeds_and_save_to_disk(path_template, path_target,
                                                                          number_of_recordings=number_of_recordings_each_parameter_stack,
                                                                          duration=duration, fs=fs, seed=seed,
                                                                          noise_level=n, filename_prefix='rec_allen',
                                                                          min_amp=min_amp, max_amp=max_amp, n_inh=n_inh,
                                                                          n_exc=n_exc)
    
if __name__ == '__main__':
    print('Welcome to MEArec Container Workflow')
    generate_template_library_set_0001()
    generate_recordings_set_0001()
    print('finished')
