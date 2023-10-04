# Functions for using cell models from ALLEN BRAIN ATLAS
# Link to data portal: https://celltypes.brain-map.org/data

# The functions are based on this notebook:
# https://github.com/alejoe91/MEArec/blob/master/notebooks/generate_recordings_with_allen_models.ipynb

# Please note that default parameters are generally used, but hard coding is also applied in some cases.

def return_allen_cell(cell_model_folder, dt=2 ** -5, start_T=0, end_T=1):
    """
    Load Allen cell in LFPy.
    :param cell_model_folder: path to cell model folder (str)
    :param dt: sampling period in s (float)
    :param start_T: start time of NEURON simulation in s (default 0)
    :param end_T: end time of NEURON simulation in s (default 1)
    :return: cell
    """
    import os
    import neuron
    import LFPy
    import json
    from pathlib import Path
    cell_model_folder = Path(cell_model_folder)
    cwd = os.getcwd()
    os.chdir(cell_model_folder)

    # compile mechanisms
    mod_folder = "modfiles"
    os.chdir(mod_folder)
    os.system('nrnivmodl')
    os.chdir('..')
    neuron.load_mechanisms(mod_folder)
    params = json.load(open("fit_parameters.json", 'r'))

    celsius = params["conditions"][0]["celsius"]
    reversal_potentials = params["conditions"][0]["erev"]
    v_init = params["conditions"][0]["v_init"]
    active_mechs = params["genome"]
    neuron.h.celsius = celsius

    cell_parameters = {
        'morphology': 'reconstruction.swc',
        'v_init': v_init,  # initial membrane potential
        'passive': False,  # turn on NEURONs passive mechanism for all sections
        'nsegs_method': 'lambda_f',  # spatial discretization method
        'lambda_f': 200.,  # frequency where length constants are computed
        'dt': dt,  # simulation time step size
        'tstart': start_T,  # start time of simulation, recorders start at t=0
        'tstop': end_T,  # stop simulation at 100 ms.
    }

    cell = LFPy.Cell(**cell_parameters)

    for sec in neuron.h.allsec():
        sec.insert("pas")
        sectype = sec.name().split("[")[0]
        for sec_dict in active_mechs:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                if not sec_dict["mechanism"] == "":
                    sec.insert(sec_dict["mechanism"])
                exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))

        for sec_dict in reversal_potentials:
            if sec_dict["section"] == sectype:
                # print(sectype, sec_dict)
                for key in sec_dict.keys():
                    if not key == "section":
                        exec("sec.{} = {}".format(key, sec_dict[key]))

    os.chdir(cwd)

    return cell


def generating_template_of_allen_models(models_path, export_filename, probe_name, cell_types, seed=0):
    """
    Generate template library of allen models using default parameters.
    :param models_path: Path to allen models (str).
    :param export_filename: Filename for template libary to be exported.
    :param probe_name: Name of probe. Probe.yaml has to be defined in anaconda3/envs/MeaRecWork/lib/python3.10/site-packages/MEAutility/electrodes (str).
    :param cell_types: Dictionary for cell models and their cell_type. Cell models has to be in models_path and have the same model_id. Example: {'489932682': 'spiny', '489932435': 'aspiny'}.
    :param seed: Defining seed to ensure reproducibility (int) (default: 0).
    :return: tempgen: MEArec TemplateGenerator object.
    """
    import numpy as np
    from pathlib import Path
    import MEArec as mr
    import MEAutility as mu
    template_params = mr.get_default_templates_params()
    template_params['seed'] = seed
    template_params['rot'] = '3drot'
    template_params['xlim'] = [0, 20] # decrease in future to [0, 10]
    template_params['n'] = 500 # 10
    print(mu.return_mea_list())
    template_params['probe'] = probe_name
    print('Probe:', template_params['probe'])

    # Generating EAPs for all cell models and assembling a template library
    # Looping through all available cell models and build a template library. Using provided cell_types.
    cell_models = [p for p in Path(models_path).iterdir()]
    print('cell_models:')
    print(cell_models)

    print('cell_models with cell_types:')
    print(cell_types)

    # Let's initialize some variables that will contain our EAPs, locations, rotations, and cell_types:
    templates, template_locations, template_rotations, template_celltypes = [], [], [], []

    for cell in cell_models:
        eaps, locs, rots = mr.simulate_templates_one_cell(cell, intra_save_folder='allen_sim',
                                                          # here: cell_folder changed to cell
                                                          params=template_params, verbose=True,
                                                          custom_return_cell_function=return_allen_cell)
        # find cell type
        cell_type = None
        for k, v in cell_types.items():
            if k in str(cell):
                cell_type = v
                break
        print("Cell", cell, "is", cell_type)

        # if first cell, initialize the arrays
        if len(templates) == 0:
            templates = eaps
            template_locations = locs
            template_rotations = rots
            template_celltypes = np.array([cell_type] * len(eaps))
        else:
            templates = np.vstack((templates, eaps))
            template_locations = np.vstack((template_locations, locs))
            template_rotations = np.vstack((template_rotations, rots))
            template_celltypes = np.concatenate((template_celltypes, np.array([cell_type] * len(eaps))))

    print('templates.shape:', templates.shape)
    print('template_locations.shape:', template_locations.shape)
    print('template_rotations.shape:', template_rotations.shape)
    print('template_celltypes.shape:', template_celltypes.shape)

    # Building a TemplateGenerator object that can be stored as a .h5 file and used to simulate recordings.
    # Creation of two dictionaries, temp_dict and info, containing the templates and related information.
    temp_dict = {'templates': templates,
                 'locations': template_locations,
                 'rotations': template_rotations,
                 'celltypes': template_celltypes}
    info = {}
    info['params'] = template_params
    info['electrodes'] = mu.return_mea_info(template_params['probe'])

    # Instantiate a TemplateGenerator object and saving .h5 file to disk.
    tempgen = mr.TemplateGenerator(temp_dict=temp_dict, info=info)
    mr.save_template_generator(tempgen=tempgen, filename=export_filename)

    return tempgen

def generate_multiple_recordings_with_different_seeds_and_save_to_disk(path_template, path_target, filename_prefix='recording_allen', number_of_recordings=1, seed=10000, verbose=True, **kwargs):
    """
    Generate and save multiple MEArec recordings of previously stored MEArec template library (TemplateGenerator object) with varying seeds.
    Function counts up the seed value and uses it for all recording parameters seeds. Please note hard coded parameters.
    :param path_template: Path to stored MEArec template library (str).
    :param path_target: Path to folder, where the recordings to be exported (str).
    :param filename_prefix: Prefix of recording filename (str) (default: 'recording_allen')
    :param number_of_recordings: Number of recordings to be generated (int) (default: 1).
    :param seed: Starting seed for recording parameters 'spiketrains', 'templates', 'convolution', 'noise' for reproducibility (int) (default: 10000).
    :param verbose: Verbose flag to suppress some prints (bool) (default: True)
    :param kwargs: Keyword arguments for recording parameters 'spiketrains', 'templates', 'convolution', 'noise'.
    :return: None
    """
    import MEArec as mr
    import os
    import pprint
    rec_params = mr.get_default_recordings_params()

    for key in kwargs:
        print(f"Key: {key}, Value {kwargs[key]}")
        if key in rec_params["spiketrains"]:
            rec_params['spiketrains'][key] = kwargs[key]
        elif key in rec_params["cell_types"]:
            rec_params['cell_types'][key] = kwargs[key]
        elif key in rec_params["templates"]:
            rec_params['templates'][key] = kwargs[key]
        elif key in rec_params['recordings']:
            rec_params['recordings'][key] = kwargs[key]
        elif key in rec_params['seeds']:
            rec_params['seeds'][key] = kwargs[key]
        else:
            print(f"Warning: {key} with Value {kwargs[key]} was not set")

    # hard overwriting
    rec_params['templates']['min_dist'] = 5  # 5 in µm
    rec_params['cell_types'] = {'excitatory': ['spiny'], 'inhibitory': ['aspiny']}
    rec_params['recordings']['filter'] = False

    # setting seeds to value of seed
    seed_spiketrain = seed  # 0
    seed_templates = seed  # 0
    seed_convolution = seed  # 0
    seed_noise = seed  # 0

    i = 0
    while i < number_of_recordings:
        rec_params['seeds']['spiketrains'] = seed_spiketrain + i
        rec_params['seeds']['templates'] = seed_templates + i
        rec_params['seeds']['convolution'] = seed_convolution + i
        rec_params['seeds']['noise'] = seed_noise + i
        i = i + 1
        if verbose:
            pprint(rec_params)
        recgen = mr.gen_recordings(params=rec_params, templates=path_template, verbose=verbose)

        # building filename
        seed_spiketrain_string = str(rec_params['seeds']['spiketrains'])
        seed_templates_string = str(rec_params['seeds']['templates'])
        seed_convolution_string = str(rec_params['seeds']['convolution'])
        seed_noise_string = str(rec_params['seeds']['noise'])

        fs_string = str(rec_params['recordings']['fs'])

        duration_string = str(rec_params['spiketrains']['duration'])
        n_exc_string = str(rec_params['spiketrains']['n_exc'])
        n_inh_string = str(rec_params['spiketrains']['n_inh'])

        noise_level_string = str(rec_params['recordings']['noise_level'])
        min_amplitude_string = str(rec_params['templates']['min_amp'])
        max_amplitude_string = str(rec_params['templates']['max_amp'])

        appendix = '.h5'

        filename = ''.join(
            [filename_prefix, '_', fs_string, '_', duration_string, '_', n_exc_string, '_', n_inh_string, '_', seed_spiketrain_string,
             '_',
             seed_templates_string, '_', seed_convolution_string, '_', seed_noise_string, '_', noise_level_string, '_',
             min_amplitude_string, '_', max_amplitude_string, appendix])

        filename_final = os.path.join(path_target, filename)

        # saving generated recordings in .h5 format
        mr.save_recording_generator(recgen=recgen, filename=filename_final)
        if verbose:
            print('saved to disk:', filename_final)
    return None

