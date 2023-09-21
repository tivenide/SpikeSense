

def read_axion_mea_csv(path_to_file, well_name=None):
    """
    Reads data from a CSV file in Axion Multiwell MEA format.

    :param path_to_file: The file path to the CSV file.
    :param well_name: (Optional) The name of the well to filter data for.

    :return: If well_name (string) is provided, returns a tuple containing the data for the specified well and the time column.
             If well_name is None, returns a tuple containing the entire DataFrame (excluding the time column) and the time column.
    """
    import pandas as pd
    df = pd.read_csv(path_to_file, sep=",")
    t = df.filter(regex='Time')
    if well_name is not None:
        well = df.filter(regex=well_name)
        return well, t
    else:
        return df.drop(columns=t.columns), t


def fit_axion_well_for_algorithms(well, t):
    import numpy as np
    well_nd = np.array(well)
    well_mikro = well_nd * 1_000_000
    well_mikro_round = well_mikro.round(4)
    timestamps = np.array(t)
    return well_mikro_round, timestamps

def fit_labview_dat_for_algorithms(data):
    import numpy as np
    data_np = np.array(data)
    timestamps_ms = data_np[:, 0]
    timestamps = np.round(timestamps_ms / 1000, 6)
    signal_raw = data_np[:, 1:]
    return signal_raw, timestamps