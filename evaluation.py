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

def calculate_metrics_for_different_thresholds(pos_gt, pos_det, plot=False):
    """
    Calculates and plots (optionally) TruePositive (TP), FalsePositive (FP) and FalseNegative (FN) with different
    assignment thresholds between two timeseries vectors. The function helps to identify the right threshold, when the
    detected timepoint does not match exactly the ground timepoint.
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
        plt.xlabel('Threshold')
        plt.ylabel('Count')
        plt.title('Assignment to metric values for different thresholds')
        plt.grid(True)
        plt.legend()
        plt.show()

    return tps, fps, fns, thresholds

