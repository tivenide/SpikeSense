
def generate_random_timeseries(duration, fs):
    import numpy as np

    num_samples = duration * fs
    time = np.arange(num_samples) / fs
    # Generate random noise signal
    noise = np.random.randn(num_samples)
    # Generate sinusoidal signal at a specific frequency
    frequency = 2000  # Frequency in Hz
    sinusoid = np.sin(2 * np.pi * frequency * time)
    # Combine noise and sinusoid
    signal = noise + sinusoid

    return time, signal


def butter_bandpass_filter_sos(data, fs, lowcut=300, highcut=6000, order=4):
    from scipy.signal import butter, sosfilt
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    sos = butter(order, [low, high], btype='band', analog=False, output='sos')
    filtered_data = sosfilt(sos, data)
    return filtered_data


def getting_timepoints_of_interest_from_threshold(electrode_data, timestamps, threshold, factor_user_pos=None, factor_user_neg=None):
    """
    Checks if a threshold value is raised by user defined factor.
    Parameters
    ----------
    electrode_data : numpy.ndarray
        The data to analyze.
    timestamps: numpy.ndarray
        Corresponding timestamps vector for data.
    threshold : float
        Value to be multiplied by user defined factor.
    factor_user_pos : float, optional
        User defined factor for positive threshold.
    factor_user_neg : float, optional
        User defined factor for negative threshold.
    Returns
    -------
    numpy.ndarray
        An array which contains timepoints, when threshold was raised.
    """
    import numpy as np
    timepoints_of_interest = []
    if factor_user_pos is not None and factor_user_neg is None:
        th_pos = threshold * factor_user_pos
        th_neg = None
    elif factor_user_pos is None and factor_user_neg is not None:
        th_pos = None
        th_neg = -threshold * factor_user_neg
    elif factor_user_pos is not None and factor_user_neg is not None:
        th_pos = threshold * factor_user_pos
        th_neg = -threshold * factor_user_neg
    else:
        raise ValueError('At least one threshold (pos or neg) has to been chosen')

    for i in range(len(electrode_data)):
        if (th_pos is not None and electrode_data[i] > th_pos) or (th_neg is not None and electrode_data[i] < th_neg):
            timepoints_of_interest.append(timestamps[i])
    return np.array(timepoints_of_interest)


def clean_timepoints_of_interest(timepoints_of_interest, refractory_period):
    """
    cleaning timepoints within refactory period.
    Parameters
    ----------
    timepoints_of_interest: np.array
        An array which contains timepoints, when threshold was raised.
    refractory_period: float
        Time in which multiple occurring spikes will be cleaned (deleted)
    Returns
    -------
    numpy.ndarray
        A cleaned version on an array which contains timepoints, when threshold was raised.
    """
    # function for cleaning timepoints within refactory period.
    import numpy as np
    if timepoints_of_interest.size > 0:
        timepoints_of_interest_cleaned = [timepoints_of_interest[0]]
        n = len(timepoints_of_interest)
        for i in range(1, n):
            if timepoints_of_interest[i] - timepoints_of_interest_cleaned[-1] > refractory_period:
                timepoints_of_interest_cleaned.append(timepoints_of_interest[i])
        timepoints_of_interest_cleaned = np.array(timepoints_of_interest_cleaned)

        return timepoints_of_interest_cleaned
    else:
        return timepoints_of_interest

def application_of_threshold_algorithm_quiroga(signal_raw, timestamps, factor_pos=None, factor_neg=3.5, refractory_period=0.001, fs=10000, lowcut=300, highcut=4500, order=4, verbose=False):
    import numpy as np
    print(f'Spikedetection with quiroga threshold calculation started')
    if verbose:
        print(f'factor_neg:\t{factor_neg}\tfactor_pos:\t{factor_pos}\trefractory_period:\t{refractory_period} sec')
        print(f'fs:\t{fs} Hz\tlowcut:\t{lowcut} Hz\thighcut:\t{highcut} Hz\torder:\t{order}')
    spiketrains = []
    sum = []
    for i in range(signal_raw.shape[1]):
        if verbose:
            print('current electrode index:', i)
        electrode_index = i
        electrode_data = signal_raw[:, electrode_index]
        if verbose:
            print('--- filtering')
        electrode_data_filtered = butter_bandpass_filter_sos(data=electrode_data, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
        if verbose:
            print('--- calculation of threshold')
        threshold = np.median(abs(electrode_data_filtered)/0.6745)
        if verbose:
            print('--- detection of spikes')
        timepoints_of_interest = getting_timepoints_of_interest_from_threshold(electrode_data_filtered, timestamps, threshold, factor_pos, factor_neg)
        if verbose:
            print('--- cleaning refactory period')
        timepoints_of_interest_cleaned = clean_timepoints_of_interest(timepoints_of_interest, refractory_period=refractory_period)
        spiketrains.append(timepoints_of_interest_cleaned)
        sum.append(timepoints_of_interest_cleaned.size)
    total_sum = np.sum(sum)
    print(f'Total detected spikes:\t{total_sum}')
    return np.array(spiketrains, dtype=object)

def plot_spiketrain_on_electrode_data(timestamps, electrode_data, spiketrain, color_spike='red', spiketrain_gt=None, color_spike_gt='green'):
    """
    Plots timepoints of interest on top of a total timeseries.
    
    Args:
        timestamps (nd.array): The timestamps of the total timeseries data.
        electrode_data (nd.array): The values of the total timeseries data.
        spiketrain (nd.array): The timepoints of interest to be plotted.
    """
    import matplotlib.pyplot as plt
    # Plot the total timeseries
    plt.plot(timestamps, electrode_data)
    
    # Plot thresholds
    #plt.plot(timestamps, np.linspace(th_neg, th_neg, len(timestamps)), color='gray')
    #plt.plot(timestamps, np.linspace(th_pos, th_pos, len(timestamps)), color='gray')

    # Plot the spiketrain
    for timepoint in spiketrain:
        value = electrode_data[timestamps == timepoint]
        plt.scatter(timepoint, value, c=color_spike)

    # Plot the spiketrain (ground truth)
    if spiketrain_gt is not None:
        for timepoint_gt in spiketrain_gt:
            value_gt = electrode_data[timestamps == timepoint_gt]
            plt.scatter(timepoint_gt, value_gt, c=color_spike_gt, marker='^')

    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude [µV]')
    plt.show()
    
def plot_spiketrains_over_time(timestamps, spiketrain, marker_size=10, color_spike='red', spiketrain_gt=None, color_spike_gt='green'):
    """
    Plots spiketrains in kind of temporal raster plot. Optional ground truth spiketrain.

    Args:
        timestamps (nd.array): The timestamps of the total timeseries data.
        spiketrain (nd.array): The timepoints of interest to be plotted. (in standard format)
        marker_size (int): Size of the marker. Default: 10.
        color_spike (str): Color of the marker for the spiketrain (detected). Default: 'red'.
        spiketrain_gt (nd.array): The timepoints of interest to be plotted (ground truth). Default: None.
        color_spike_gt (str): Color of the marker for the spiketrain (ground truth). Default: 'green'.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    num_arrays = len(spiketrain)
    y_values = np.arange(1, num_arrays + 1)

    plt.figure(figsize=(10, 6))
    for i, array in enumerate(spiketrain):
        x_values = np.array(array)
        y_values_i = np.full_like(x_values, y_values[i])
        plt.scatter(x_values, y_values_i, label=f'{i+1}', marker='o', color=color_spike, s=marker_size)

    if spiketrain_gt is not None:
        for i_gt, array_gt in enumerate(spiketrain_gt):
            x_values_gt = np.array(array_gt)
            y_values_i_gt = np.full_like(x_values_gt, y_values[i_gt])
            plt.scatter(x_values_gt, y_values_i_gt, label=f'{i + 1}', marker='^', color=color_spike_gt, s=marker_size)

    plt.xlabel('Time [sec]')
    plt.ylabel('Electrode')
    plt.yticks(y_values, [f'{i+1}' for i in range(num_arrays)])
    plt.title('Spiketrain')
    #plt.legend()
    plt.grid(True)
    plt.xlim(-0.005, max(timestamps)+0.005)
    plt.show()

def plot_spiketrain_over_time(timestamps, spiketrain, line_length=1.00, line_width=2, color_spike='black', y_label='Electrode'):
    """
    Plots spiketrain in kind of temporal raster plot with vertical lines.

    Args:
        timestamps (nd.array): The timestamps of the total timeseries data.
        spiketrain (nd.array): The timepoints of interest to be plotted. (in standard format)
        line_length (float): Length of vertical line. Default: 1.00
        line_width (int): Width of vertical line. Default: 2.
        color_spike (str): Color of the marker for the spiketrain. Default: 'black'.
        y_label (str): Label of y-axis. Default: 'Electrode'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    num_arrays = len(spiketrain)
    y_values = np.arange(1, num_arrays + 1)

    plt.figure(figsize=(10, 6))
    for i, array in enumerate(spiketrain):
        y_value = y_values[i]
        for time in array:
            plt.vlines(time, y_value - line_length/2, y_value + line_length/2, colors=color_spike, linewidth=line_width)

    plt.xlabel('Time [sec]')
    plt.ylabel(y_label)
    plt.yticks(y_values, [f'{i+1}' for i in range(num_arrays)])
    plt.title('Spiketrain')
    plt.grid(True)
    plt.xlim(-0.005, max(timestamps)+0.005)
    plt.ylim(0.5, num_arrays + 0.5)
    plt.show()

def plot_spiketrains_over_time_above_electrode_data(timestamps, electrode_data, spiketrain_1, spiketrain_2, spiketrain_3):
    """
    Plots spiketrains with vertical lines above one electrode data timeseries. Uses subplots with synchronized time axis.

    Args:
        timestamps (nd.array): The timestamps of the total timeseries data.
        electrode_data(nd.array): The values of the total timeseries data of one electrode.
        spiketrain_1 (nd.array): The timepoints of interest to be plotted of one electrode. Usually ground truth timepoints.
        spiketrain_2 (nd.array): The timepoints of interest to be plotted of one electrode. Usually first detection algorithm.
        spiketrain_3 (nd.array): The timepoints of interest to be plotted of one electrode. Usually second detection algorithm.

    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(4, 1, height_ratios=[0.2, 0.2, 0.2, 2], hspace=0.05)

    # Plot the electrode data in the fourth subplot first
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(timestamps, electrode_data)
    ax4.set_xlabel('Time in sec')
    ax4.set_ylabel('Voltage in µV')

    # Plot the spiketrains above electrode data
    ax1 = fig.add_subplot(gs[0], sharex=ax4)
    ax1.set_title('Comparison of spiketrains')
    ax1.set_ylabel('GT')
    ax1.set_yticks([])
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for time in spiketrain_1:
        ax1.axvline(x=time, color='black', linestyle='-')

    ax2 = fig.add_subplot(gs[1], sharex=ax4)
    ax2.set_ylabel('Det 1')
    ax2.set_yticks([])
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for time in spiketrain_2:
        ax2.axvline(x=time, color='red', linestyle='--')

    ax3 = fig.add_subplot(gs[2], sharex=ax4)
    ax3.set_ylabel('Det 2')
    ax3.set_yticks([])
    ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    for time in spiketrain_3:
        ax3.axvline(x=time, color='red', linestyle='--')

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.1)

    # Variable to prevent retriggering of the xlim_changed event
    xlim_changed_triggered = False

    # Adjusting the axis limits and zoom behavior for all subplots
    def on_xlims_change(ax):
        global xlim_changed_triggered
        if not xlim_changed_triggered:
            xlim_changed_triggered = True
            for subplot in [ax1, ax2, ax3, ax4]:
                subplot.set_xlim(ax.get_xlim())
        xlim_changed_triggered = False

    ax4.callbacks.connect('xlim_changed', on_xlims_change)

    plt.show()