
def save_spiketrains_to_hdf5(input_array, filename):
    import h5py
    with h5py.File(filename, 'w') as file:
        st_group = file.create_group('st')
        for i, values in enumerate(input_array):
            electrode_name = f'electrode_{i}'
            if len(values) > 0:
                st_group.create_dataset(electrode_name, data=values)
            else:
                # If values is an empty list or array, create an empty dataset
                st_group.create_dataset(electrode_name, shape=(0,))

        # Optional:
        # st_group.attrs['description'] = 'electrode values'

        print(f"Data saved to {filename} successfully.")

def read_spiketrains_from_hdf5(filename):
    import h5py
    import numpy as np
    data = []
    with h5py.File(filename, 'r') as file:
        st_group = file['st']
        electrode_keys = sorted(st_group.keys(), key=lambda x: int(x.split('_')[1]))
        electrodes = len(electrode_keys)
        data = np.empty(electrodes, dtype=object)
        for i, electrode_key in enumerate(electrode_keys):
            dataset = st_group[electrode_key]
            if dataset.shape[0] > 0:
                data[i] = dataset[:]
            else:
                data[i] = np.array([])
    return data


def extract_timestamps(input_array):
    """
    extract timestamps of preprocessed array
    :param input_array: input array with dimensions (electrodes, 3, timestamps) with
        second dimension: (0: values, 1: labels, 2: timestamps)
    :return: spiketrains in standard array format
    """
    import numpy as np
    sensors = input_array.shape[0]
    output_array = []

    for i in range(sensors):
        labels = input_array[i, 1]
        timestamps = input_array[i, 2]

        # Extract timestamps where label is 1
        timestamps_1 = [timestamp for timestamp, label in zip(timestamps, labels) if label == 1]

        if len(timestamps_1) > 0:
            output_array.append(np.array(timestamps_1))
        else:
            output_array.append(np.array([]))

    return np.array(output_array, dtype=object)

def get_ground_truth_spiketrains_in_standard_array_format(path, threshold=100):
    """
    Preprocessing pipeline for one recording (without windowing) and returns ground truth spiketrains.
    :param path: path to recording file
    :param threshold: The maximum distance between an electrode location and a neuron location for them
            to be considered a match.
    :return: spiketrains in standard array format
    """
    from utilities import prepro
    any = 'any'
    prepro_obj = prepro(path_to_original_recordings=any, path_to_target_prepro_results=any, threshold=any, windowsize_in_sec=any, step_size=any, feature_calculation=any, scaler_type=any, sampling_strategy=any)
    signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = prepro_obj.import_recording_h5(path=path)
    labels_of_all_spiketrains = prepro_obj.create_labels_of_all_spiketrains(ground_truth, timestamps)
    assignments = prepro_obj.assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, threshold=threshold)
    merged_data = prepro_obj.merge_data_to_location_assignments(assignments, signal_raw.transpose(), labels_of_all_spiketrains, timestamps)
    output = extract_timestamps(merged_data)
    return output


def calculate_nearest_distances(gt, det):
    """
    Calculates the distances to the nearest timepoints between the two input arrays.
    :param gt: sorted array with ground truth timepoints (minimum size: 2 items)
    :param det: sorted array with detected timepoints (minimum size: 2 items)
    :return: distances with length of gt
    """
    import numpy as np
    distances = []
    for gt_time in gt:
        # old code:
        #nearest_distance = np.inf
        #for det_time in det:
        #    distance = abs(gt_time - det_time)
        #    if distance < nearest_distance:
        #        nearest_distance = distance
        # new code:
        distances_tmp = np.abs(np.subtract(det, gt_time))
        nearest_distance = np.min(distances_tmp)
        distances.append(nearest_distance)
    return distances

def calculate_metrics_for_different_gt_temporal_assignment_thresholds(pos_gt, pos_det, plot=False):
    """
    Calculates and plots (optionally) TruePositive (TP), FalsePositive (FP) and FalseNegative (FN) with different
    assignment thresholds between two timeseries vectors. The function helps to identify the right threshold, when the
    detected timepoint does not match exactly the ground truth timepoint.
    :param pos_gt: sorted array with ground truth timepoints of the positive class (minimum size: 2 items)
    :param pos_det: sorted array with detected timepoints of the positive class (minimum size: 2 items)
    :param plot: boolean. Default: False
    :return: tps, fps, fns, thresholds
    """
    import numpy as np
    import matplotlib.pyplot as plt
    tps = []
    fps = []
    fns = []
    pos_distances = calculate_nearest_distances(gt=pos_gt, det=pos_det)
    thresholds = np.linspace(0, np.max(pos_distances), num=100)
    for threshold in thresholds:
        tp = np.sum(pos_distances <= threshold)
        tps.append(tp)
        fp = len(pos_det) - tp
        fps.append(fp)
        fn = len(pos_gt) - tp
        fns.append(fn)

    if plot:
        plt.plot(thresholds, tps, label='TPs')
        plt.plot(thresholds, fps, label='FPs')
        plt.plot(thresholds, fns, label='FNs')
        plt.xlabel('Threshold in sec')
        plt.ylabel('Count')
        plt.title('TPs, FPs, FNs in dependency of gt_temporal_assignment_thresholds')
        plt.grid(True)
        plt.legend()
        plt.show()

    return tps, fps, fns, thresholds

def calculate_metrics_for_mea_recording(gt_spiketrains, det_spiketrains, threshold=0.002, verbose=False):
    """
    Calculates TruePositive (TP), FalsePositive (FP) and FalseNegative (FN) for mea-recording
    :param gt_spiketrains: ground truth spiketrains (positive class)
    :param det_spiketrains: detection spiketrains (positive class)
    :param threshold: temporal distance threshold for true positive (tp) assignment. If this is set too high,
        it can lead to negative values in other metrics, as these are derived from tp. Default: 0.002 sec
    :param verbose: boolean. Prints results per electrode. Default: False
    :return: Summation of all TPs, FPs and FNs (total_tps, total_fps, total_fns)
    """
    import numpy as np
    print('calculate metrics for mea chip')

    tps = []
    fps = []
    fns = []
    tns = []
    gts = []
    dets = []
    for i in range(gt_spiketrains.shape[0]):
        gt_spiketrain = gt_spiketrains[i]
        det_spiketrain = det_spiketrains[i]
        gts.append(gt_spiketrain.size)
        dets.append(det_spiketrain.size)
        if gt_spiketrain.size == 0 and det_spiketrain.size >=0:
            fp = det_spiketrain.size
            fps.append(fp)
            if verbose:
                print(f'el:{i}\tfp:{det_spiketrain.size}')
        elif gt_spiketrain.size >= 0 and det_spiketrain.size == 0:
            fn = gt_spiketrain.size
            fns.append(fn)
            if verbose:
                print(f'el:{i}\tfn:{gt_spiketrain.size}')
        elif gt_spiketrain.size == 0 and det_spiketrain.size == 0:
            tn = 'not calculated' # sample_frequency * recording_duration
            if verbose:
                print(f'el:{i}\ttn:{tn}')
        elif gt_spiketrain.size >= 0 and det_spiketrain.size >= 0:
            if verbose:
                print(f'el:{i}\there we have to calculate')
            if gt_spiketrain.size >= 2 and det_spiketrain.size >= 2:
                if verbose:
                    print(f'el:{i}\there we can calculate')
                pos_distances = calculate_nearest_distances(gt=gt_spiketrain, det=det_spiketrain)
                tp = np.sum(np.array(pos_distances) <= threshold)
                fp = det_spiketrain.size - tp
                fn = gt_spiketrain.size - tp
                if verbose:
                    print(f'el:{i}\ttp:{tp}\tfp:{fp}\tfn:{fn}')
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
    total_tps = np.sum(np.array(tps))
    total_fps = np.sum(np.array(fps))
    total_fns = np.sum(np.array(fns))
    total_gt = np.sum(np.array(gts))
    total_det = np.sum(np.array(dets))
    print('totals:')
    print(f'tp: {total_tps}\tfp: {total_fps}\tfn:{total_fns}')
    print(f'gt: {total_gt}\tdet: {total_det}')
    return total_tps, total_fps, total_fns


def calculate_metrics_for_different_gt_location_assignment_thresholds(path_to_h5_gt_spiketrains, path_to_h5_det_spiketrains, threshold=0.002):
    """
    Calculates and plots TruePositive (TP), FalsePositive (FP) and FalseNegative (FN) with different neuron-electrode
    assignment thresholds between two spiketrains sets. The function helps to identify the right threshold.
    :param path_to_h5_gt_spiketrains: path to ground truth spiketrains in standard array format
    :param path_to_h5_det_spiketrains: path to detection spiketrains in standard array format
    :param threshold: temporal distance threshold for true positive (tp) assignment. If this is set too high,
        it can lead to negative values in other metrics, as these are derived from tp. Default: 0.002 sec
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    tps = []
    fps = []
    fns = []
    thresholds = np.linspace(start=10, stop=200, num=10)

    for gt_location_assignment_threshold in thresholds:
        gt_spiketrains = get_ground_truth_spiketrains_in_standard_array_format(path_to_h5_gt_spiketrains, threshold=gt_location_assignment_threshold)
        det_spiketrains = read_spiketrains_from_hdf5(path_to_h5_det_spiketrains)
        total_tps, total_fps, total_fns = calculate_metrics_for_mea_recording(gt_spiketrains=gt_spiketrains, det_spiketrains=det_spiketrains, threshold=threshold)
        tps.append(total_tps)
        fps.append(total_fps)
        fns.append(total_fns)

    plt.plot(thresholds, tps, label='TPs')
    plt.plot(thresholds, fps, label='FPs')
    plt.plot(thresholds, fns, label='FNs')

    plt.xlabel('Threshold in µm')
    plt.ylabel('Count')
    plt.title('TPs, FPs, FNs in dependency of gt_location_assignment_threshold')
    plt.grid(True)
    plt.legend()
    plt.show()

    return tps, fps, fns, thresholds

# SNR calculation

def calculate_estimated_snr_for_mea_recording(signal_raw):
    import numpy as np
    peaks = []
    #stds = []
    #mad_quirogas = []

    for i in range(signal_raw.shape[1]):
        electrode_data = signal_raw[:,i]
        peak = np.max(abs(electrode_data))
        #std = np.std(electrode_data)
        #mad_quiroga = np.median(abs(electrode_data)/0.6745)
        peaks.append(peak)
        #stds.append(std)
        #mad_quirogas.append(mad_quiroga)

    mad_quiroga_total = np.median(abs(signal_raw)/0.6745)

    snr = np.mean(peaks) / mad_quiroga_total
    print(f'estimated snr: {(snr):>0.2f}')
    return snr

def calculate_snr_for_mea_recording(signal_raw, signal_raw_noise):
    import numpy as np
    snrs = []
    snrs_dB = []
    for i in range(signal_raw.shape[1]):
        electrode_data_signal = signal_raw[:, i]
        electrode_data_noise = signal_raw_noise[:, i]
        electrode_snr = np.sqrt(np.mean(electrode_data_signal**2)) / np.sqrt(np.mean(electrode_data_noise**2))
        snrs.append(electrode_snr)

        electrode_snr_dB = 10 * np.log10(electrode_snr)
        snrs_dB.append(electrode_snr_dB)

    snr = np.mean(snrs)
    snr_dB = np.mean(snrs_dB)
    print(f'snr: {(snr):>0.2f}')
    print(f'snr: {(snr_dB):>0.2f} dB')
    return snr, snrs, snr_dB, snrs_dB

def get_window_size_in_index_count(timestamps, window_size_in_sec):
    """
    calculate window size in index counts from defined windowsize (in sec)
    :param timestamps: all timestamps (used for calculation)
    :param window_size_in_sec: windowsize in seconds
    :return: window_size_in_count
    """
    def count_indexes_up_to_value(arr, value):
        import numpy as np
        # Find the indexes where the array values are less than or equal to the specified value
        indexes = np.where(arr <= value)[0]
        # Count the number of indexes
        count = len(indexes)
        return count

    window_size_in_count = count_indexes_up_to_value(timestamps, window_size_in_sec)
    return window_size_in_count - 1

def cut_windows_from_timeseries(raw_in, timestamps_in, times_in, window_range_in):
    """
    Cuts out windows from a time series at specified interesting timepoints.

    Parameters:
    - raw_in: ndarray, input time series data of shape (raw_length, ).
    - timestamps_in: ndarray, timestamps corresponding to raw_in of shape (raw_length, ).
    - times_in: ndarray, interesting timepoints within the time series of shape (times_length, ).
    - window_range_in: int, the number of timesteps before and after each interesting timepoint.

    Returns:
    - output_vector: ndarray, a 2D array containing the windows of raw values of shape (times_length, window_length).
    """
    import numpy as np
    output_vector = []
    #window_length = 2 * window_range_in + 1

    for time_point in times_in:
        # Find the index of the closest timestamp to the time_point
        closest_idx = np.argmin(np.abs(timestamps_in - time_point))

        # Calculate the start and end indices for the window
        start_idx = max(0, closest_idx - window_range_in)
        end_idx = min(len(raw_in), closest_idx + window_range_in + 1)

        # Extract the window from raw_in
        window = raw_in[start_idx:end_idx]

        # Calculate the padding size on both sides
        #left_padding = max(0, window_length - len(window)) // 2
        #right_padding = max(0, window_length - len(window) - left_padding)

        # Pad the window with zeros on both sides
        #window = np.pad(window, (left_padding, right_padding), mode='constant')

        output_vector.append(window)

    return np.array(output_vector, dtype=object)

def calculate_snr_for_mea_recording_with_ground_truth_spiketrain(signal_raw, timestamps, st_gt, window_range_in_sec=0.002):
    '''
    Calculates SNR for MEA recording with ground truth spiketrains.
    The function uses the spiketrains input to extract the windows with spikes per electrode. Then absolute maximum of
    a window is calculated and stored in a temporary vector. The median of this vector is used to calculate the SNR
    with standard deviation of the raw signal per electrode. Finally, the median of all electrode SNR values is
    calculated for resulting SNR value of MEA recording file.
    :param signal_raw: array of size (number_total_timestamps, number_total_electrodes) which contains the raw values
    :param timestamps: array of size (number_total_timestamps, ) which contains all corresponding timestamps
    :param st_gt: spiketrains in standard array format (ground truth). Array of size (number_total_electrodes,)
    - not all electrodes contains spikes.
    :param window_range_in_sec: defines the positive and negative span around the ground truth time point
    :return: snr
    '''
    import numpy as np

    snrs = []
    for i in range(st_gt.shape[0]):
        if st_gt[i].size == 0:
            pass
        elif st_gt[i].size >= 0:
            range_in = get_window_size_in_index_count(timestamps, window_range_in_sec)
            windows = cut_windows_from_timeseries(raw_in=signal_raw[:, i], timestamps_in=timestamps, times_in=st_gt[i],
                                                  window_range_in=range_in)

            spike_abs_values = []
            for n in range(0, len(windows)):
                abs_value = abs(windows[n]).max()
                spike_abs_values.append(abs_value)

            median_spike_abs_value = np.median(spike_abs_values)
            #mean_spike_abs_value = np.mean(spike_abs_values)
            #peak_spike_abs_value = np.max(spike_abs_values)
            std_value = np.std(signal_raw[:, i])

            snr = median_spike_abs_value / std_value
            snrs.append(snr)
    print(f'calculated snr: {(np.median(snrs)):>0.2f}')
    return np.median(snrs)


# plotting metrics after training for inspection

def calc_mean_loss_over_epochs(loss, epochs_num):
    import numpy as np
    step_width = int(len(loss) / epochs_num)
    loss_windows = []
    means = []
    for i in range(0, len(loss) - step_width + 1, step_width):
        loss_windows.append(loss[i:i + step_width])

    for i in range(0, len(loss_windows)):
        mean = np.mean(loss_windows[i])
        means.append(mean)
    return means


def plot_losses_smoothed_over_epochs(loss_train, loss_eval, epochs_num):
    import matplotlib.pyplot as plt
    loss_train_means = calc_mean_loss_over_epochs(loss=loss_train, epochs_num=epochs_num)
    loss_eval_means = calc_mean_loss_over_epochs(loss=loss_eval, epochs_num=epochs_num)

    epochs = range(1, epochs_num + 1)
    plt.plot(epochs, loss_train_means, 'b', label='Training loss')
    plt.plot(epochs, loss_eval_means, 'r', label='Evaluation loss')
    plt.xlim(1, epochs_num)

    # plt.plot(loss_train_means, 'b', label='Training Loss')
    # plt.plot(loss_eval_means, 'r', label='Evaluation Loss')
    plt.title('Training and evaluation losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_metrics_over_epochs_v2(metrics, title='Metrics over Epochs'):
    import matplotlib.pyplot as plt
    marker_size = 2
    epochs = range(1, len(metrics) + 1)
    f1_scores = [metric['f1'] for metric in metrics]
    precision = [metric['precision'] for metric in metrics]
    recall = [metric['recall'] for metric in metrics]

    plt.plot(epochs, f1_scores, label='F1-Score', marker='o', markersize=marker_size)
    plt.plot(epochs, precision, label='Precision', marker='o', markersize=marker_size)
    plt.plot(epochs, recall, label='Recall', marker='o', markersize=marker_size)

    plt.ylim(0, 1)
    plt.xlim(1, len(metrics))

    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve_v2(precision, recall, baseline_factor=None):
    """

    :param precision:
    :param recall:
    :param baseline_factor: factor, which describes the random line in plot. Typically calculated by Positive Class / (Positive Class + Negative Class)
    :return:
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    pr_auc = auc(recall, precision)

    plt.figure(figsize=(5, 5))

    plt.plot(recall, precision, color='b', label=f'PR Curve (AUC = {pr_auc:.4f})')

    if baseline_factor is not None:
        plt.plot([0, 1], [baseline_factor, baseline_factor], color='r', linestyle='--',
                 label=f'Random (AUC = {baseline_factor:.4f})')

    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve')
    if baseline_factor is not None:
        plt.legend()
    plt.show()


def plot_roc_curve_v2(fpr, tpr, auc):
    import matplotlib.pyplot as plt
    # import numpy as np
    # gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    # ix = np.argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='r', linestyle='--', label='Random (AUC = 0.50)')
    # plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label=f'Best (gmeans: {gmeans[ix]:.2f})')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()


def pipeline_for_basic_evaluation_plots_of_training_process(file_path_to_dataset_h5, baseline_factor):
    """
    A basic pipeline for:
        - loading metrics from saved dataset.h5 in standard format
        - plotting f1, precision, recall over epochs for training and evaluation
        - plotting loss for training and evaluation
        - plotting roc and pr-curve for test
    :param file_path_to_dataset_h5: file to saved dataset.h5 in standard format
    :param baseline_factor: baseline factor for pr-curve
    :return: none
    """
    from utilities import read_data_from_h5
    data_obj = read_data_from_h5(file_path=file_path_to_dataset_h5)

    # loading metrics
    metrics_train = data_obj.read_metrics_from_h5(mode='train')
    metrics_eval = data_obj.read_metrics_from_h5(mode='eval')
    metrics_test = data_obj.read_metrics_test_from_h5()

    # plotting f1, precision, recall over epochs
    plot_metrics_over_epochs_v2(metrics_train, title='Metrics over epochs (training)')
    plot_metrics_over_epochs_v2(metrics_eval, title='Metrics over epochs (evaluation)')

    # loading and plotting loss
    loss_train, loss_eval = data_obj.read_metrics_losses_from_h5()
    epochs_num = data_obj.read_metrics_get_epochs_count()
    plot_losses_smoothed_over_epochs(loss_train=loss_train, loss_eval=loss_eval, epochs_num=epochs_num)

    # plotting roc and pr-curve
    prc = metrics_test['pr_curve']['precision']
    rcl = metrics_test['pr_curve']['recall']

    fpr = metrics_test['roc_curve_1']['fpr']
    tpr = metrics_test['roc_curve_1']['tpr']
    auc = metrics_test['roc_curve_1']['auc']
    plot_precision_recall_curve_v2(prc, rcl, baseline_factor=baseline_factor)
    plot_roc_curve_v2(fpr, tpr, auc)

