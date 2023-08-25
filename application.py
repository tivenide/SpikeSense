def application():
    print('SpikeSense application started')

    path_to_mearec_h5 = f'mearec_recording_file.h5'
    path_to_model = f'work_dir_tm_selu1_lr_1e-5/best_TransformerModel.pth'
    path_to_results_npy = f'st_results_tm.npy'
    path_to_results_h5 = f'st_results_tm.h5'
    device = f'cuda:0'

    from utilities import prepro
    prepro_object = prepro(path_to_original_recordings='', path_to_target_prepro_results='', threshold=10, windowsize_in_sec=0.001, step_size=None, feature_calculation=False, scaler_type='standard', sampling_strategy=0.5) # here: parameters just for initialization
    signal_raw, timestamps, ground_truth, channel_positions, template_locations = prepro_object.import_recording_h5(path=path_to_mearec_h5)

    import torch
    from ai_modeling import using
    from custom_models import TransformerModel

    loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=12, num_heads=8, dropout=0.1)
    loaded_model.load_state_dict(torch.load(path_to_model))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    #loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model, device=device)

    st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps, window_size_in_sec=0.002)

    from utilities import save_frame_to_disk
    save_frame_to_disk(frame=st, path_target=path_to_results_npy)

    from evaluation import save_spiketrains_to_hdf5
    save_spiketrains_to_hdf5(input_array=st, filename=path_to_results_h5)

application()