def application_quick(importer, output_path):
    print('SpikeSense application started (quick)')

    signal_raw = importer.import_signal_raw()
    timestamps = importer.import_timestamps()

    path_to_model = f'quick/working_directory/best_DenseModel.pth'
    path_to_results_npy = f'{output_path}_st_fnn.npy'
    path_to_results_h5 = f'{output_path}_st_fnn.h5'

    import torch
    from ml_framework import using
    from ml_models import DenseModel

    loaded_model = DenseModel(in_features=25, hidden_features=50, out_features=2)
    loaded_model.load_state_dict(torch.load(path_to_model))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    #loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model)

    st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps, window_size_in_sec=0.002)

    from utilities import save_frame_to_disk
    save_frame_to_disk(frame=st, path_target=path_to_results_npy)

    from evaluation import save_spiketrains_to_hdf5
    save_spiketrains_to_hdf5(input_array=st, filename=path_to_results_h5)