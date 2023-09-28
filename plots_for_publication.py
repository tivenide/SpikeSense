# Caution: Resource intensive functions!
# Hard coded functions, requires appropriate file structures!

# different sample frequencies

def create_spiketrains_for_different_sample_frequencies():
    import torch
    from utilities import prepro
    from utilities import save_frame_to_disk
    from utilities import load_frame_from_disk
    from evaluation import save_spiketrains_to_hdf5
    from ai_modeling import using
    from custom_models import TransformerModel

    loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=6, num_heads=8, dropout=0.1)
    loaded_model.load_state_dict(torch.load('work_dir_tm_hyper/tm_selu1_0-1_64_8_6_32_1e-4_1e-5_under_0-8/best_TransformerModel.pth'))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    # loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model, device='cuda')

    different_fs = [5000, 8000, 10000, 12000, 12500, 16000]
    for i in different_fs:
        print('current fs:', i)
        path_to_mearec_h5 = f'tm_evaluation_different_fs/source/rec_allen_fs{i}_60_17_12_2000_2000_2000_2000_10_20_32.h5'

        prepro_object = prepro(path_to_original_recordings='', path_to_target_prepro_results='', threshold='', windowsize_in_sec='', step_size='', feature_calculation='', scaler_type='', sampling_strategy='') # here: parameters just for initialization
        signal_raw, timestamps, ground_truth, channel_positions, template_locations = prepro_object.import_recording_h5(path=path_to_mearec_h5)

        st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps)

        save_frame_to_disk(st, f'tm_evaluation_different_fs/st_transformer_eval_ds_fs_{i}.npy')
        frame = load_frame_from_disk(f'tm_evaluation_different_fs/st_transformer_eval_ds_fs_{i}.npy')

        save_spiketrains_to_hdf5(input_array=frame, filename=f'tm_evaluation_different_fs/st_transformer_eval_ds_fs_{i}.h5')

def calculate_and_plot_evaluation_results_on_different_sample_frequencies():
    import matplotlib.pyplot as plt
    from evaluation import read_spiketrains_from_hdf5
    from evaluation import get_ground_truth_spiketrains_in_standard_array_format
    from evaluation import calculate_metrics_for_mea_recording

    prc = []
    rcl = []
    f1 = []

    different_fs = [5000, 8000, 10000, 12000, 12500, 16000]
    for i in different_fs:
        print('current fs:', i)
        path_to_mearec_h5 = f'tm_evaluation_different_fs/source/rec_allen_fs{i}_60_17_12_2000_2000_2000_2000_10_20_32.h5'
        st = read_spiketrains_from_hdf5(filename=f'tm_evaluation_different_fs/st_transformer_eval_ds_fs_{i}.h5')

        st_gt = get_ground_truth_spiketrains_in_standard_array_format(path=path_to_mearec_h5)
        total_tps, total_fps, total_fns = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st)

        precision = total_tps / (total_tps + total_fps)
        recall = total_tps / (total_tps + total_fns)
        f1_score = total_tps / (total_tps + 0.5 * (total_fps + total_fns))

        prc.append(precision)
        rcl.append(recall)
        f1.append(f1_score)

        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1_score)

    different_fs_kilohertz = [val / 1000 for val in different_fs]

    marker_size = 2
    plt.plot(different_fs_kilohertz, f1, marker='o', label='F1-Score', markersize=marker_size)
    plt.plot(different_fs_kilohertz, prc, marker='o', label='Precision', markersize=marker_size)
    plt.plot(different_fs_kilohertz, rcl, marker='o', label='Recall', markersize=marker_size)
    plt.ylim(0, 1)
    plt.xlim(min(different_fs_kilohertz), max(different_fs_kilohertz))
    plt.xlabel('Sample Frequency in kHz')
    plt.ylabel('Score')
    # plt.title()
    plt.legend()
    plt.grid(True)
    plt.show()


# different window sizes

def create_spiketrains_for_different_window_sizes():
    import torch
    from ai_modeling import using
    from custom_models import TransformerModel
    from utilities import prepro
    from utilities import save_frame_to_disk
    from evaluation import save_spiketrains_to_hdf5

    path_to_mearec_h5 = f'tm_evaluation_different_win_sizes/rec_allen_fs10000_60_17_12_2000_2000_2000_2000_10_20_32.h5'
    path_to_model = f'work_dir_tm_hyper/tm_selu1_0-1_64_8_6_32_1e-4_1e-5_under_0-8/best_TransformerModel.pth'

    prepro_object = prepro(path_to_original_recordings='', path_to_target_prepro_results='', threshold='', windowsize_in_sec='', step_size='', feature_calculation='', scaler_type='', sampling_strategy='') # here: parameters just for initialization
    signal_raw, timestamps, ground_truth, channel_positions, template_locations = prepro_object.import_recording_h5(path=path_to_mearec_h5)

    loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=6, num_heads=8, dropout=0.1)
    loaded_model.load_state_dict(torch.load(path_to_model))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    #loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model, device='cuda')

    for i in [0.0010, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024, 0.0026, 0.0028, 0.0030]:
        print('current windowsize:', i)
        path_to_results_npy = f'tm_evaluation_different_win_sizes/st_transformer_eval_ds_fs_10000_winsize_{i}.npy'
        path_to_results_h5 = f'tm_evaluation_different_win_sizes/st_transformer_eval_ds_fs_10000_winsize_{i}.h5'

        st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps, window_size_in_sec=i)

        save_frame_to_disk(frame=st, path_target=path_to_results_npy)
        save_spiketrains_to_hdf5(input_array=st, filename=path_to_results_h5)


def calculate_and_plot_evaluation_results_on_different_window_sizes():
    import matplotlib.pyplot as plt

    from evaluation import get_ground_truth_spiketrains_in_standard_array_format
    from evaluation import read_spiketrains_from_hdf5
    from evaluation import calculate_metrics_for_mea_recording

    path_to_mearec_h5 = f'tm_evaluation_different_win_sizes/rec_allen_fs10000_60_17_12_2000_2000_2000_2000_10_20_32.h5'
    st_gt = get_ground_truth_spiketrains_in_standard_array_format(path=path_to_mearec_h5)

    prc = []
    rcl = []
    f1 = []

    # calculate
    different_window_sizes = [0.0010, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0022, 0.0024, 0.0026, 0.0028, 0.0030]
    for i in different_window_sizes:
        print('current windowsize:', i)

        st = read_spiketrains_from_hdf5(filename=f'tm_evaluation_different_win_sizes/st_transformer_eval_ds_fs_10000_winsize_{i}.h5')

        total_tps, total_fps, total_fns = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st)

        precision = total_tps / (total_tps + total_fps)
        recall = total_tps / (total_tps + total_fns)
        f1_score = total_tps / (total_tps + 0.5 * (total_fps + total_fns))

        prc.append(precision)
        rcl.append(recall)
        f1.append(f1_score)

        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1_score)

    # plot
    marker_size = 2
    different_window_sizes_milliseconds = [val * 1000 for val in different_window_sizes]

    plt.plot(different_window_sizes_milliseconds, f1, marker='o', label='F1-Score', markersize=marker_size)
    plt.plot(different_window_sizes_milliseconds, prc, marker='o', label='Precision', markersize=marker_size)
    plt.plot(different_window_sizes_milliseconds, rcl, marker='o', label='Recall', markersize=marker_size)
    plt.ylim(0, 1)
    plt.xlim(min(different_window_sizes_milliseconds), max(different_window_sizes_milliseconds))
    plt.xlabel('Window size in ms')
    plt.ylabel('Score')
    #plt.title()
    plt.legend()
    plt.grid(True)
    plt.show()

# different noise levels

def create_spiketrains_for_different_noise_levels():
    import torch
    from utilities import prepro
    from utilities import save_frame_to_disk
    from evaluation import save_spiketrains_to_hdf5
    from ai_modeling import using
    from custom_models import TransformerModel, DenseModel

    path_to_model = f'work_dir_tm_hyper/tm_selu1_0-1_64_8_6_32_1e-4_1e-5_under_0-8/best_TransformerModel.pth'
    #path_to_model = f'work_dir_hyper_other_nets/under_0-8/FNN/best_DenseModel.pth'

    loaded_model = TransformerModel(input_dim=1, hidden_size=64, num_classes=2, num_layers=6, num_heads=8, dropout=0.1)
    #loaded_model = DenseModel(in_features=20, hidden_features=50, out_features=2)

    loaded_model.load_state_dict(torch.load(path_to_model))
    # see tutorial: https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    # loaded_model.to(device)
    usr_object = using(loaded_model=loaded_model, device='cuda')

    different_noise_levels = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]

    for i in different_noise_levels:
        print('current nl:', i)

        path_to_mearec_h5 = f'tm_evaluation_different_nl/source/rec_allen_fs10000_60_17_12_2000_2000_2000_2000_{i}_20_32.h5'
        path_to_results_npy = f'tm_evaluation_different_nl/st_transformer_eval_ds_fs_10000_{i}_20_32.npy'
        path_to_results_h5 = f'tm_evaluation_different_nl/st_transformer_eval_ds_fs_10000_{i}_20_32.h5'
        #path_to_results_npy = f'tm_evaluation_different_nl/st_fnn_eval_ds_fs_10000_{i}_20_32.npy'
        #path_to_results_h5 = f'tm_evaluation_different_nl/st_fnn_eval_ds_fs_10000_{i}_20_32.h5'

        prepro_object = prepro(path_to_original_recordings='', path_to_target_prepro_results='', threshold='', windowsize_in_sec='', step_size='', feature_calculation='', scaler_type='', sampling_strategy='')  # here: parameters just for initialization
        signal_raw, timestamps, ground_truth, channel_positions, template_locations = prepro_object.import_recording_h5(path=path_to_mearec_h5)

        st = usr_object.application_of_model(signal_raw=signal_raw, timestamps=timestamps, window_size_in_sec=0.002)

        save_frame_to_disk(frame=st, path_target=path_to_results_npy)
        save_spiketrains_to_hdf5(input_array=st, filename=path_to_results_h5)


def calculate_and_plot_f1_scores_of_different_methods_for_different_noise_levels():
    import matplotlib.pyplot as plt
    from evaluation import read_spiketrains_from_hdf5
    from evaluation import get_ground_truth_spiketrains_in_standard_array_format
    from evaluation import calculate_metrics_for_mea_recording
    from evaluation import calculate_estimated_snr_for_mea_recording
    from evaluation import calculate_snr_for_mea_recording_with_ground_truth_spiketrain
    from utilities import prepro
    from custom_algorithms import application_of_threshold_algorithm_quiroga

    prc = []
    rcl = []
    f1 = []

    f1_th = []
    f1_th_2 = []
    f1_fnn = []

    different_noise_levels = [5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
    snrs = []
    for i in different_noise_levels:
        print('current nl:', i)

        path_to_mearec_h5 = f'tm_evaluation_different_nl/source/rec_allen_fs10000_60_17_12_2000_2000_2000_2000_{i}_20_32.h5'
        st_gt = get_ground_truth_spiketrains_in_standard_array_format(path=path_to_mearec_h5)

        # noise level estimation
        any = 'any'
        prepro_obj = prepro(path_to_original_recordings=any, path_to_target_prepro_results=any, threshold=any,
                            windowsize_in_sec=any, step_size=any, feature_calculation=any, scaler_type=any,
                            sampling_strategy=any)
        signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = prepro_obj.import_recording_h5(path=path_to_mearec_h5)
        calculate_estimated_snr_for_mea_recording(signal_raw=signal_raw)
        snr = calculate_snr_for_mea_recording_with_ground_truth_spiketrain(signal_raw=signal_raw, timestamps=timestamps, st_gt=st_gt)
        snrs.append(snr)

        # Transformer model
        st = read_spiketrains_from_hdf5(filename=f'tm_evaluation_different_nl/st_transformer_eval_ds_fs_10000_{i}_20_32.h5')
        total_tps, total_fps, total_fns = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st)

        precision = total_tps / (total_tps + total_fps)
        recall = total_tps / (total_tps + total_fns)
        f1_score = total_tps / (total_tps + 0.5 * (total_fps + total_fns))

        prc.append(precision)
        rcl.append(recall)
        f1.append(f1_score)

        print('Precision:', precision)
        print('Recall:', recall)
        print('F1-Score:', f1_score)


        low_cut = 300
        high_cut = 4500
        # Threshold algorithm
        st_th, th_pos, th_neg = application_of_threshold_algorithm_quiroga(signal_raw=signal_raw, timestamps=timestamps, factor_pos=4.0, factor_neg=4.0, lowcut=low_cut, highcut=high_cut)
        total_tps_th, total_fps_th, total_fns_th = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st_th)
        f1_score_th = total_tps_th / (total_tps_th + 0.5 * (total_fps_th + total_fns_th))
        f1_th.append(f1_score_th)

        # Threshold algorithm
        st_th_2, th_pos_2, th_neg_2 = application_of_threshold_algorithm_quiroga(signal_raw=signal_raw, timestamps=timestamps, factor_pos=5.0, factor_neg=5.0, lowcut=low_cut, highcut=high_cut)
        total_tps_th_2, total_fps_th_2, total_fns_th_2 = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st_th_2)
        f1_score_th_2 = total_tps_th_2 / (total_tps_th_2 + 0.5 * (total_fps_th_2 + total_fns_th_2))
        f1_th_2.append(f1_score_th_2)

        # FNN model
        st_fnn = read_spiketrains_from_hdf5(filename=f'tm_evaluation_different_nl/st_fnn_eval_ds_fs_10000_{i}_20_32.h5')
        total_tps_fnn, total_fps_fnn, total_fns_fnn = calculate_metrics_for_mea_recording(gt_spiketrains=st_gt, det_spiketrains=st_fnn)
        f1_score_fnn = total_tps_fnn / (total_tps_fnn + 0.5 * (total_fps_fnn + total_fns_fnn))
        f1_fnn.append(f1_score_fnn)

    # plot

    marker_size = 2
    plt.plot(snrs, f1, marker='o', label='TM', color='blue', markersize=marker_size)
    plt.plot(snrs, f1_fnn, marker='o', label='FNN', color='dodgerblue', markersize=marker_size)
    plt.plot(snrs, f1_th, marker='o', label='TH (fac=4.0)', color='red', markersize=marker_size)
    plt.plot(snrs, f1_th_2, marker='o', label='TH (fac=5.0)', color='darkred', markersize=marker_size)

    plt.xlim(min(snrs), max(snrs))
    plt.ylim(0, 1)
    plt.xlabel('SNR')
    plt.ylabel('Score')
    plt.title('F1-Score for different detection methods')
    plt.legend()
    plt.grid(True)
    plt.show()


