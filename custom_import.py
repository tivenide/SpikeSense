def import_dat_return_data(path_to_dat_file:str):
    """
        Imports existing .dat file from LabView Software and fit data in pd.Dataframe.
        Parameters
        ----------
        path_to_dat_file : string
            Path to .dat file.
        Returns
        -------
        data_2 : pd.Dataframe
            contains voltage values of the electrodes to timestamp
        meta : list
            contains meta information about imported file
        names_formatted_list : list
            contains formatted column headers for pd.dataframe
        error_val : string
            contains error notice of width of the data
    """
    import pandas as pd
    import fnmatch
    #Read meta data and format it into a list
    i = pd.read_csv(path_to_dat_file , sep="\s+", encoding="cp1252", nrows=0)
    meta = list(i.columns.values)
    #Read time and amplitude data for checking
    data = pd.read_csv(path_to_dat_file, sep="\s+", encoding="cp1252", skiprows=3, nrows=6)
    #Read names of electrodes and format into list
    names_unformatted = pd.read_csv(path_to_dat_file, sep="\t", encoding="cp1252", skiprows=[0,1], nrows=1)
    names_unformatted_list = list(names_unformatted.columns.values)
    names_formatted_list=[]
    my_pattern = "Unnamed:*"
    for name_to_check in names_unformatted_list:
        if not fnmatch.fnmatch(name_to_check, my_pattern):
            j = name_to_check.replace(' ','')
            names_formatted_list.append(j)
    #Check if length of names and length of data is equal
    len_of_names_formatted_list = len(names_formatted_list)
    len_of_data = len(data.columns)
    error_var = "Length of names fits length of data"
    if len_of_names_formatted_list != len_of_data:
        error_var = "Length of names doesn't fit length of data: " + str(len_of_names_formatted_list) + " != " + str(len_of_data)
    #Read time and amplitude data. Fitting formatted names into data and remove units
    data_2 = pd.read_csv(path_to_dat_file, sep="\s+", encoding="cp1252", names = names_formatted_list, skiprows=4)
    rec_dur = data_2.iloc[:, 0].max() # Recording Duration
    meta.append(rec_dur)
    #Return values
    print(".dat file successfully imported and into pd.dataframe formatted")
    return data_2, meta, names_formatted_list, error_var

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