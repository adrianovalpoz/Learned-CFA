"""
Adapted from R code also available in INESCTEC repository 'simultaneous-prediction-intervals'
Created By  : Miguel Ribeiro - luis.m.ribeiro@inesctec.pt   Line 3
Created Date: 07/04/2022

"""

import numpy as np
import pandas as pd


def chebyshev_dist(day_scenarios, mean_val, type='simple', sd_val=np.nan):
    """
    calculate chebyshev distances by columns

    :param day_scenarios: dataframe with timestep in rows and scenarios in columns
    :param mean_val:  array of mean value by timestep
    :param type: type of distance: simple, weighted, signed
    :param sd_val: array of standard deviation by timestep
    :return: list of distances
    """

    dist = []
    for column in day_scenarios.columns:
        if type == 'simple':
            # max value of absolute difference between column and mean array
            _dist = [(abs(day_scenarios[column] - mean_val)).max()]
            dist = dist + _dist
        elif type == 'weighted':
            # max value of absolute difference between column and mean array
            # divided by sd array
            _dist = [(abs(day_scenarios[column] - mean_val) / sd_val).max()]
            dist = dist + _dist
        elif type == 'signed':
            # max value of absolute difference between column and mean array
            # divided by sd array
            vmax = (abs(day_scenarios[column] - mean_val) / sd_val).max()
            # location of max value
            tmax = np.where((abs(day_scenarios[column] - mean_val) / sd_val) == vmax)[0][0]
            # sign of the difference of column value with index tmax and the
            # mean value with index tmax,
            # multiplied by vmax
            _dist = [np.sign(day_scenarios[column][tmax] - mean_val[tmax]) * vmax]
            dist = dist + _dist

    return dist


def chebyshev_interval(coverage, day_scenarios):
    """
    Calculate min and max values (bands) for each timestamp

    :param coverage: caverage band
    :param day_scenarios: dataframe with timestep in rows and scenarios in
     columns
    :return:define min and max values (bands) for each timestamp
    """

    mean_val = day_scenarios.mean(axis=1)  # mean value by timestamp
    sd_val = day_scenarios.std(axis=1)  # standard deviation by timestamp

    # run chebyshev_distance function
    distances = chebyshev_dist(day_scenarios=day_scenarios, mean_val=mean_val,
                               type='weighted', sd_val=sd_val)

    # number of cenarios to be used to established the bands, defined by the
    # coverage value
    M = int(coverage * len(day_scenarios.columns))

    # order by ascending order the number of distances defined by M
    idx_order_distance = np.argsort(distances)[0:M]

    # order the scenarios by idx_order_distance
    sample = day_scenarios.iloc[:, idx_order_distance]

    # define min and max values (bands) for each timestamp
    bands = pd.DataFrame({'min': sample.min(axis=1),
                          'max': sample.max(axis=1)})

    return bands
