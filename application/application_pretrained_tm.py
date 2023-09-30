def application_pretrained_tm(importer, output_path):
    print('SpikeSense application started')

    signal_raw = importer.import_signal_raw()
    timestamps = importer.import_timestamps()

    path_to_model = f'pretrained_models/best_TransformerModel.pth'
    path_to_results_npy = f'{output_path}_st_pretrained_tm.npy'
    path_to_results_h5 = f'{output_path}_st_pretrained_tm.h5'

    import torch
    from ml_framework import using
    from ml_models import TransformerModel
    loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=6, num_heads=8, dropout=0.1)
    loaded_model.load_state_dict(torch.load(path_to_model))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    #loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model)

    st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps, window_size_in_sec=0.002)

    from utilities import save_frame_to_disk
    save_frame_to_disk(frame=st, path_target=path_to_results_npy)

    from evaluation import save_spiketrains_to_hdf5
    save_spiketrains_to_hdf5(input_array=st, filename=path_to_results_h5)