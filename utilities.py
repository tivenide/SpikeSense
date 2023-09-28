# save to and load from disk
def save_frame_to_disk(frame, path_target):
    import numpy as np
    print('started saving frame to disk')
    np.save(path_target, frame, allow_pickle=True)
    print('frame saved to disk')


def load_frame_from_disk(path_source):
    import numpy as np
    print('started loading frame from disk')
    frame = np.load(path_source, allow_pickle=True)
    print('frame loaded from disk')
    return frame

class prepro():
    def __init__(self, path_to_original_recordings, path_to_target_prepro_results, threshold, windowsize_in_sec, step_size, feature_calculation, scaler_type, sampling_strategy):
        self.path_to_original_recordings = path_to_original_recordings
        self.path_to_target_prepro_results = path_to_target_prepro_results
        self.threshold = threshold
        self.windowsize_in_sec = windowsize_in_sec
        self.step_size = step_size
        self.feature_calculation = feature_calculation
        self.scaler_type = scaler_type
        self.sampling_strategy = sampling_strategy

        self.x_train_res = None
        self.y_train_res = None
        self.x_test_crp = None
        self.y_test_crp = None
        self.x_val_crp = None
        self.y_val_crp = None

    def run(self):
        frame = self.preprocessing_for_multiple_recordings(path=self.path_to_original_recordings, threshold=self.threshold, window_size_in_sec=self.windowsize_in_sec, step_size=self.step_size, feature_calculation=self.feature_calculation)
        if self.scaler_type is not None:
            frame = self.normalize_frame(frame, scaler_type=self.scaler_type)

        self.x_train_res, self.y_train_res, self.x_test_crp, self.y_test_crp, self.x_val_crp, self.y_val_crp = self.dataset_pipeline_for_training_process(frame=frame, sampling_strategy=self.sampling_strategy, test_and_val_size=0.4, val_size_of_test_and_val_size=0.5, verbose= True)
        self.saving_prepro_results_in_h5(self.path_to_target_prepro_results)

    def saving_prepro_results_in_h5(self, file_path):
        import h5py
        with h5py.File(file_path, 'w') as file:
            # Save hyperparameters
            prepro_group = file.create_group('prepro_meta')
            prepro_group.attrs['path_to_original_recordings'] = self.path_to_original_recordings
            prepro_group.attrs['threshold'] = self.threshold
            prepro_group.attrs['windowsize_in_sec'] = self.windowsize_in_sec
            #file.attrs['step_size'] = self.step_size
            prepro_group.attrs['feature_calculation'] = self.feature_calculation
            #file.attrs['scaler_type'] = self.scaler_type
            prepro_group.attrs['sampling_strategy'] = self.sampling_strategy

            # Handle the None case for step_size
            if self.step_size is not None:
                prepro_group.attrs['step_size'] = self.step_size
            else:
                prepro_group.attrs['step_size'] = "None"

            # Handle the None case for scaler_type
            if self.scaler_type is not None:
                prepro_group.attrs['scaler_type'] = self.scaler_type
            else:
                prepro_group.attrs['scaler_type'] = "None"



            # Save datasets as separate datasets
            datasets_group = file.create_group('datasets')
            datasets_group.create_dataset('x_train', data=self.x_train_res)
            datasets_group.create_dataset('y_train', data=self.y_train_res)
            datasets_group.create_dataset('x_test', data=self.x_test_crp)
            datasets_group.create_dataset('y_test', data=self.y_test_crp)
            datasets_group.create_dataset('x_val', data=self.x_val_crp)
            datasets_group.create_dataset('y_val', data=self.y_val_crp)

    def import_recording_h5(self, path):
        """
        Import recording h5 file from MEArec.
        :param path: path to file
        :return: signal_raw, timestamps, ground_truth, channel_positions, template_locations
        """
        import h5py  # hdf5
        import numpy as np
        h5 = h5py.File(path, 'r')
        signal_raw = np.array(h5["recordings"])
        timestamps = np.array(h5["timestamps"])
        ground_truth = []
        for i in range(len(h5["spiketrains"].keys())):
            ground_truth.append(np.array(h5["spiketrains"][str(i)]["times"]))
        channel_positions = np.array(h5["channel_positions"])  # indexes of columns x: 1 y: 2 z: 0
        template_locations = np.array(h5["template_locations"])  # indexes of columns x: 1 y: 2 z: 0
        return signal_raw, timestamps, ground_truth, channel_positions, template_locations

    def create_labels_for_spiketrain(self, timestamps, times):
        """
        Assign ground truth label of times to the nearest value of timestamps.
        :param timestamps: all timestamps
        :param times: ground truth timestamps of occurring spikes
        :return: labels: Returns list of length of timestamps with 1s at positions of times and 0s at the other positions.
        """
        import bisect
        import numpy as np
        labels = np.zeros(len(timestamps), dtype=int)
        times_sorted = np.sort(timestamps)
        for i, t in enumerate(times):
            index = bisect.bisect_left(times_sorted, t)
            if index == 0:
                nearest_timestamp = times_sorted[0]
            elif index == len(times_sorted):
                nearest_timestamp = times_sorted[-1]
            else:
                left_timestamp = times_sorted[index - 1]
                right_timestamp = times_sorted[index]
                if t - left_timestamp < right_timestamp - t:
                    nearest_timestamp = left_timestamp
                else:
                    nearest_timestamp = right_timestamp
            nearest_index = np.searchsorted(timestamps, nearest_timestamp)
            labels[nearest_index] = 1
        return labels

    def create_labels_of_all_spiketrains(self, ground_truth, timestamps):
        """
        Create labels for all ground_truth spiketrains using create_labels_for_spiketrain()
        :param ground_truth:
        :param timestamps:
        :return: labels_of_all_spiketrains: Returns numpy array of all ground_truth spiketrains with 1s for a spike and
            0s otherwise.
        """
        import numpy as np
        labels_of_all_spiketrains = []
        for i in range(len(ground_truth)):
            labels = self.create_labels_for_spiketrain(timestamps, ground_truth[i])
            labels_of_all_spiketrains.append(labels)
        return np.array(labels_of_all_spiketrains)

    def assign_neuron_locations_to_electrode_locations(self, electrode_locations, neuron_locations, threshold):
        """
        Assigns the index of a neuron location to the index of an electrode location if
        the distance between them is less than or equal to the threshold value.
        :param electrode_locations:
        :param neuron_locations:
        :param threshold: The maximum distance between an electrode location and a neuron location for them
            to be considered a match.
        :return:
        """
        import pandas as pd
        import numpy as np

        electrode_locations_df = pd.DataFrame(electrode_locations)
        neuron_locations_df = pd.DataFrame(neuron_locations)

        # Compute the distance between each electrode location and each neuron location
        distances = np.sqrt(
            ((electrode_locations_df.values[:, np.newaxis, :] - neuron_locations_df.values) ** 2).sum(axis=2))

        # Create an empty DataFrame to store the results
        assignments = pd.DataFrame(index=electrode_locations_df.index, columns=neuron_locations_df.index, dtype=bool)

        # Assign each channel position to its closest neuron_locations (if within the threshold distance)
        for i, point_idx in enumerate(neuron_locations_df.index):
            mask = distances[:, i] <= threshold
            assignments.iloc[:, i] = mask

        return assignments

    def merge_data_to_location_assignments(self, assignments, signal_raw, labels_of_all_spiketrains, timestamps):
        """
        Assigns the label vectors to the raw data. For the merging of multiple spiketrains to one electrode the
        np.logical_or() is used. For electrodes without an assignment to spiketrains empty spiketrains are generated.
        Additionally, timestamps are added.
        :param assignments: A DataFrame representing the local assignment between neurons and electrodes.
            With rows corresponding to electrodes and columns corresponding to neurons. Each cell in the
            DataFrame is True if the corresponding channel position is within the threshold distance of the
            corresponding neuron, and False otherwise. If a channel position is not assigned to any neuron position,
            the corresponding cells are False.
        :param signal_raw: A numpy array representing the recorded signal, with rows
            corresponding to electrodes of the MEA and columns corresponding to timestamps.
        :param labels_of_all_spiketrains: A numpy array representing the labels, with rows
            corresponding to spiketrains of the different neurons and columns corresponding to timestamps.
        :param timestamps:
        :return: merged_data: A numpy array representing the merged data. It's build like nested lists. The structure:
            [[[raw_data of the first electrode],[labels of the first electrode],[timestamps of the first electrode]],
            [[raw_data of the second electrode],[labels of the second electrode],[timestamps of the second electrode]], etc.]
        """
        import numpy as np

        assignments2 = np.array(assignments, dtype=bool)
        merged_data = []

        for i in range(assignments2.shape[0]):  # iterate over rows in assignments
            row = assignments2[i]  # equals electrode
            merged = np.zeros(len(labels_of_all_spiketrains[0]))  # generating empty spiketrains
            for j, value in enumerate(row):  # iterate over "columns" in rows
                if value:
                    merged = np.logical_or(merged, labels_of_all_spiketrains[j, :])
            merged_data.append([signal_raw[i], merged.astype(int), timestamps])
        return np.array(merged_data)

    def devide_3_vectors_into_equal_windows_with_step(self, x1, x2, x3, window_size, step_size=None):
        """
        Devides vectors x1, x2, x3 into windows with one window_size. step_size is used to generate more windows with overlap.
        :param x1: Input list to be devided.
        :param x2: Input list to be devided.
        :param x3: Input list to be devided.
        :param window_size: Size of each window.
        :param step_size: If the step_size is not provided, it defaults to the window_size.
            If the step_size is set to True, it is set to half of the window_size.
            If the step_size is set to any other value, it is used directly as the step_size.
        :return: Returns for every input a list of lists. Each included list represents a window.
        """
        if step_size is None:
            step_size = window_size
        elif step_size is True:
            step_size = window_size // 2
        x1_windows = []
        x2_windows = []
        x3_windows = []
        for i in range(0, len(x1) - window_size + 1, step_size):
            x1_windows.append(x1[i:i + window_size])
            x2_windows.append(x2[i:i + window_size])
            x3_windows.append(x3[i:i + window_size])
        return x1_windows, x2_windows, x3_windows

    def application_of_windowing_v2(self, merged_data, window_size, step_size=None, feature_calculation=False):
        """
        Version 2 of application_of_windowing(). Currently designed just for one window_size and one step_size. Devides
        merged data into windows and calculate features for windows. Additionally, set a label for each window.
        :param merged_data: A numpy array representing the merged data from merge_data_to_location_assignments().
        :param window_size: Size of each window in counts.
        :param step_size: Size (in counts) of offset between windows.
            If the step_size is not provided, it defaults to the window_size.
            If the step_size is set to True, it is set to half of the window_size.
            If the step_size is set to any other value, it is used directly as the step_size.
        :param feature_calculation: bool. Defines the boolean value if calculate_features() is active or not.
        :return: A custom ndarray representing the merged data, which is devided by windows. With rows corresponding to windows
            and columns corresponding to signal_raw (signals), labels, timestamps, electrode number, features and label_per_window.
        """
        import numpy as np

        if step_size is None:
            step_size = window_size
        elif step_size is True:
            step_size = window_size // 2
        elif step_size is not None:
            step_size = int(step_size)

        # calculate number of features dynamically based on the returned feature vector from calculate_features()
        sample_data = merged_data[0][0:window_size]
        features_size = self.calculate_features(window_data=sample_data, calculate=feature_calculation).shape[0]

        # defining empty custom ndarray
        num_windows = sum((data.shape[1] - window_size) // step_size + 1 for data in merged_data)
        frame = np.zeros((num_windows,), dtype=[
            ('signals', np.float32, (window_size,)),
            ('labels', np.float32, (window_size,)),
            ('timestamps', np.float32, (window_size,)),
            ('electrode_number', np.int32),
            ('features', np.float32, (features_size,)),
            ('label_per_window', np.int32)
        ])

        curr_idx = 0
        for i, data in enumerate(merged_data):
            # calculate windows
            num_windows_i = (data.shape[1] - window_size) // step_size + 1
            win1 = np.lib.stride_tricks.as_strided(
                data[0], shape=(num_windows_i, window_size), strides=(data[0].strides[0] * step_size, data[0].strides[0]))
            win2 = np.lib.stride_tricks.as_strided(
                data[1], shape=(num_windows_i, window_size), strides=(data[1].strides[0] * step_size, data[1].strides[0]))
            win3 = np.lib.stride_tricks.as_strided(
                data[2], shape=(num_windows_i, window_size), strides=(data[2].strides[0] * step_size, data[2].strides[0]))

            for j in range(num_windows_i):
                # apply windowing to resulting frame
                frame[curr_idx]['signals'] = win1[j]
                frame[curr_idx]['labels'] = win2[j]
                frame[curr_idx]['timestamps'] = win3[j]
                frame[curr_idx]['electrode_number'] = i
                # calculate features for each window
                frame[curr_idx]['features'] = self.calculate_features(window_data=win1[j], calculate=feature_calculation)
                frame[curr_idx]['label_per_window'] = self.label_a_window_from_labels_of_a_window(win2[j])
                curr_idx += 1

        return frame

    def calculate_features(self, window_data, calculate=False):
        """
        Calculates features for input data.
        :param window_data: input data for calculation
        :param calculate: bool.
            If calculate is set to True, it calculates the defined features and returns the feature array.
            If calculate is set to False, it returns a np.zeros array.
        :return: feature array or np.zeros array with size 1
        """
        import numpy as np
        if calculate is True:
            # Assuming 3 features for example purposes
            num_features = 3
            features = np.zeros((num_features,))
            features[0] = np.mean(window_data)
            features[1] = np.min(window_data)
            features[2] = np.max(window_data)
            return features
        elif calculate is False:
            return np.zeros((1,))


    def label_a_window_from_labels_of_a_window(self, window_data):
        """
        Finds the max value of input data and returns it as integer. Input data of labels of a window should only be 0s and 1s.
        :param window_data: input data for calculation
        :return: label. It represents the label of the input window.
        """
        import numpy as np
        label = int(np.max(window_data))
        return label

    def count_indexes_up_to_value(self, arr, value):
        import numpy as np
        # Find the indexes where the array values are less than or equal to the specified value
        indexes = np.where(arr <= value)[0]
        # Count the number of indexes
        count = len(indexes)
        return count


    def get_window_size_in_index_count(self, timestamps, window_size_in_sec):
        """
        calculate window size in index counts from defined windowsize (in sec)
        :param timestamps: all timestamps (used for calculation)
        :param window_size_in_sec: windowsize in seconds
        :return: window_size_in_count
        """
        window_size_in_count = self.count_indexes_up_to_value(timestamps, window_size_in_sec)
        return window_size_in_count - 1

    def preprocessing_for_one_recording(self, path, threshold=100, window_size_in_sec=0.002, step_size=None, feature_calculation=False):
        """
        preprocessing pipeline for one recording (without normalization)
        :param path: path to recording file
        :param window_size_in_sec: window size in seconds (default = 0.001)
        :return: frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
            and columns corresponding to signal_raw, labels, timestamps and electrode number.
        """
        signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = self.import_recording_h5(path)
        labels_of_all_spiketrains = self.create_labels_of_all_spiketrains(ground_truth, timestamps)
        assignments = self.assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, threshold=threshold)
        merged_data = self.merge_data_to_location_assignments(assignments, signal_raw.transpose(), labels_of_all_spiketrains,
                                                         timestamps)
        window_size_in_counts = self.get_window_size_in_index_count(timestamps, window_size_in_sec)
        frame = self.application_of_windowing_v2(merged_data=merged_data, window_size=window_size_in_counts, step_size=step_size,
                                            feature_calculation=feature_calculation)
        print('preprocessing finished for:', path)
        return frame

    def preprocessing_for_multiple_recordings(self, path, threshold=100, window_size_in_sec=0.002, step_size=None, feature_calculation=False):
        """
        preprocessing pipeline for multiple recordings (without normalization)
        :param path: path to recording files. Only MEArec generated h5. recording files may be located here.
        :return: frame_of_multiple_recordings: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
            and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
        """
        from pathlib import Path
        import numpy as np
        recordings = [p for p in Path(path).iterdir()]
        frame_of_multiple_recordings = None
        print('preprocessing started for:', path)
        for rec in recordings:
            frame_of_one_recording = self.preprocessing_for_one_recording(path=rec, threshold=threshold, window_size_in_sec=window_size_in_sec, step_size=step_size, feature_calculation=feature_calculation)
            if frame_of_multiple_recordings is None:
                frame_of_multiple_recordings = frame_of_one_recording.copy()
            else:
                frame_of_multiple_recordings = np.hstack((frame_of_multiple_recordings, frame_of_one_recording))
        print('preprocessing finished for:', path)
        return frame_of_multiple_recordings

    def normalize_frame(self, frame, scaler_type='minmax'):
        """
        Normalizes the raw data in the input array using the specified scaler type
        :param frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
            and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
        :param scaler_type: possible Scalers from sklearn.preprocessing: StandardScaler, MinMaxScaler, RobustScaler
        :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
            and columns corresponding to signal_raw (normalized), labels, timestamps and electrode number. No assignment to the recording!
        """
        import numpy as np
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Scaler type {scaler_type} not supported. Please choose 'standard', 'minmax', or 'robust'")

        print(f"Normalization with scaler type '{scaler_type}' started")
        for i in frame:
            data_raw = i[0]
            data_norm = scaler.fit_transform(data_raw.reshape(-1, 1))
            i[0] = data_norm.flatten()
        print(f"Normalization with scaler type '{scaler_type}' finished")
        return frame


    # preparation from preprocessing to data loader
    def splitting_data_into_train_test_val_set(self, data, labels, test_and_val_size=0.4, val_size_of_test_and_val_size=0.5):
        """
        Splits data and labels into training, test and validation set.
        :param data: input set which contains data
        :param labels: input set which contains labels for data
        :param test_and_val_size: size of test and validation set combined. Rest equals training set.
        :param val_size_of_test_and_val_size: size of validation set corresponding to test_and_val_size. Rest equals test set.
        :return: training, test and validation set for data and labels
        """
        from sklearn.model_selection import train_test_split
        X = data
        y = labels
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_and_val_size, stratify=y)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size_of_test_and_val_size,
                                                        stratify=y_test)
        return x_train, y_train, x_test, y_test, x_val, y_val

    def balancing_dataset_with_undersampling(self, data, labels, sampling_strategy='majority', verbose=True):
        """
        balancing dataset with random undersampling with sampling strategy 'majority'
        :param sampling_strategy:
        :param data: input data
        :param labels: corresponding labels for input data.
        :param verbose: Using print(). Default: True.
        :return: balanced data and labels (unshuffeled)
        """
        from imblearn.under_sampling import RandomUnderSampler
        if verbose:
            print('balancing started')
        undersample = RandomUnderSampler(sampling_strategy=sampling_strategy) #'majority', tried: 0.5
        data_result, labels_result = undersample.fit_resample(data, labels)
        if verbose:
            print('balancing finished')
        return data_result, labels_result

    def cropping_dataset(self, data, labels, cropping_size):
        """
        crop dataset to size of cropping_size to a subset
        :param data: input data
        :param labels: corresponding labels for input data.
        :param cropping_size: float between 0 and 1. proportion of resulting dataset from input dataset
        :return: cropped data and cropped labels
        """
        from sklearn.model_selection import train_test_split
        X = data
        y = labels
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=cropping_size, stratify=y)
        return x_test, y_test

    def dataset_pipeline_for_training_process(self, frame, sampling_strategy='majority', test_and_val_size=0.4, val_size_of_test_and_val_size=0.5, verbose=True):
        """
        pipeline for splitting, balancing and cropping datasets for training process
        :param frame: input ndarray which contains data and corresponding labels
        :param verbose: (bool) Default True. Prints proportions.
        :return: undersampled training set, cropped test set, cropped validation set
        """
        # init
        data = frame['signals']
        labels = frame['label_per_window']
        # splitting data
        x_train, y_train, x_test, y_test, x_val, y_val = self.splitting_data_into_train_test_val_set(data=data, labels=labels, test_and_val_size=test_and_val_size, val_size_of_test_and_val_size=val_size_of_test_and_val_size)

        if verbose:
            print('frame: spikes:', labels.sum(), 'total:', len(labels))
            print('train: spikes:', y_train.sum(), 'total:', len(y_train))
            print('test: spikes:', y_test.sum(), 'total:', len(y_test))
            print('val: spikes:', y_val.sum(), 'total:', len(y_val))

        # undersample training set
        x_train_res, y_train_res = self.balancing_dataset_with_undersampling(data=x_train, labels=y_train, sampling_strategy=sampling_strategy, verbose=verbose)
        # calculation of cropping size
        spikes_per_frame = (labels.sum()) / (len(labels))
        # cropping test set
        x_test_crp, y_test_crp = self.cropping_dataset(x_test, y_test, spikes_per_frame)
        # cropping validation set
        x_val_crp, y_val_crp = self.cropping_dataset(x_val, y_val, spikes_per_frame)

        if verbose:
            print('spikes_per_frame:', spikes_per_frame)
            print('train_res: spikes:', y_train_res.sum(), 'total:', len(y_train_res))
            print('test_crp: spikes:', y_test_crp.sum(), 'total:', len(y_test_crp))
            print('val_crp: spikes:', y_val_crp.sum(), 'total:', len(y_val_crp))

        return x_train_res, y_train_res, x_test_crp, y_test_crp, x_val_crp, y_val_crp

class read_data_from_h5():
    def __init__(self, file_path):
        self.file_path = file_path

    def read_prepro_metadata_from_h5(self):
        import h5py
        model_metadata = {}
        with h5py.File(self.file_path, 'r') as file:
            if 'model_meta' in file:
                model_meta_group = file['prepro_meta']
                for key, value in model_meta_group.attrs.items():
                    model_metadata[key] = value
        return model_metadata

    def read_model_metadata_from_h5(self):
        import h5py
        model_metadata = {}
        with h5py.File(self.file_path, 'r') as file:
            if 'model_meta' in file:
                model_meta_group = file['model_meta']
                for key, value in model_meta_group.attrs.items():
                    model_metadata[key] = value
        return model_metadata

    def read_trainer_metadata_from_h5(self):
        import h5py
        model_metadata = {}
        with h5py.File(self.file_path, 'r') as file:
            if 'trainer_meta' in file:
                model_meta_group = file['trainer_meta']
                for key, value in model_meta_group.attrs.items():
                    model_metadata[key] = value
        return model_metadata


    def read_metrics_from_h5(self, mode):
        import h5py
        metrics = []
        if mode == 'train':
            group_name = 'metrics_train'
        elif mode == 'eval':
            group_name = 'metrics_evaluation'
        else:
            raise ValueError(f"Mode {mode} not supported. Please choose 'train' or 'eval'")

        with h5py.File(self.file_path, 'r') as file:
            if group_name in file:
                metrics_group = file[group_name]
                epoch_keys_sorted = sorted(metrics_group.keys(), key=lambda x: int(x.split('_')[1]))
                for i, epoch_key in enumerate(epoch_keys_sorted):
                    epoch_group = metrics_group[epoch_key]
                    metric_dict = {}
                    for key in epoch_group.keys():
                        if isinstance(epoch_group[key], h5py.Dataset):
                            metric_dict[key] = epoch_group[key][...]

                        elif isinstance(epoch_group[key], h5py.Group):
                            sub_dict = {}
                            for sub_key in epoch_group[key].keys():
                                sub_dict[sub_key] = epoch_group[key][sub_key][...]
                            metric_dict[key] = sub_dict
                        else:
                            raise ValueError(f"Unsupported data type for metric: {type(epoch_group[key])}")
                    for attr_key, attr_value in epoch_group.attrs.items():
                        if isinstance(attr_value, (int, float)):
                            metric_dict[attr_key] = attr_value

                    metrics.append(metric_dict)
        return metrics

    def read_metrics_test_from_h5(self):
        import h5py

        metrics = {}

        with h5py.File(self.file_path, 'r') as file:
            if 'metrics_test' in file:
                metric_group = file['metrics_test']
                for key in metric_group.keys():
                    if isinstance(metric_group[key], h5py.Dataset):
                        metrics[key] = metric_group[key][...]
                    elif isinstance(metric_group[key], h5py.Group):
                        sub_dict = {}
                        for sub_key in metric_group[key].keys():
                            sub_dict[sub_key] = metric_group[key][sub_key][...]
                        metrics[key] = sub_dict
                    else:
                        raise ValueError(f"Unsupported data type for metric: {type(metric_group[key])}")
                for attr_key, attr_value in metric_group.attrs.items():
                    if isinstance(attr_value, (int, float)):
                        metrics[attr_key] = attr_value

        return metrics

    def read_metrics_losses_from_h5(self):
        import h5py
        with h5py.File(self.file_path, 'r') as file:
            losses_group = file['loss']
            loss_eval = losses_group['evaluation_loss'][()]
            loss_train = losses_group['training_loss'][()]

        return loss_train, loss_eval

    def read_metrics_get_epochs_count(self):
        import h5py
        with h5py.File(self.file_path, 'r') as file:
            eval_group = file['metrics_evaluation']
            epochs_count = 0
            for epoch_key in eval_group.keys():
                epochs_count = epochs_count+1
        return epochs_count


#custom import functions
def import_recording_h5_only_signal_raw(path):
    """
    Import recording h5 file from MEArec.
    :param path: path to file
    :return: signal_raw
    """
    import h5py  # hdf5
    import numpy as np
    h5 = h5py.File(path, 'r')
    signal_raw = np.array(h5["recordings"])
    return signal_raw



def count_parameters(model):
    # from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Modelname: {model.__class__.__name__}\t\tParameters: {count}')
    return count