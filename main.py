import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import roc_curve
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import itertools
import pathlib
import os
from datetime import datetime

from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from functions import *
from data_process import *
from figures import *

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    # Load in test_predictions and subset by race
    test_predictions_a = add_labels(add_race_age(add_prescription(get_test_predictions().query('is_train == 0'))))
    test_predictions_w = get_white_data(test_predictions_a)
    test_predictions_nw = get_nonwhite_data(test_predictions_a)

    print(test_predictions_a.shape,
          test_predictions_w.shape,
          test_predictions_nw.shape)

    # Thresholds
    thresholds = dict(NIT=0.129, SXT=0.18, CIP=0.258, LVX=0.239)

    racedfs = [test_predictions_a,
               test_predictions_w,
               test_predictions_nw]

    # Recommendations for thresholds
    allpt, whitept, nonwhitept = create_rec_dfs(racedfs,
                                                thresholds)
    print(allpt.shape,
          whitept.shape,
          nonwhitept.shape)

    test_predictions_y, test_predictions_m, test_predictions_o = get_age_groups(test_predictions_a)

    agedfs = [test_predictions_y,
              test_predictions_m,
              test_predictions_o]

    youngpt, midpt, oldpt = create_rec_dfs(agedfs,
                                           thresholds)

    print(youngpt.shape,
          midpt.shape,
          oldpt.shape)

    dfs_race = [allpt, whitept, nonwhitept]
    dfs_age = [allpt, youngpt, midpt, oldpt]

    # Make the directory for figure/csv outputs
    dtime = make_fig_fold()

    thresholds_df = pd.DataFrame(thresholds, index=['Thresholds'])
    thresholds_df.to_csv(f"{dtime}/{dtime}" + "_thresholds_used.csv")

    # ECDF plots
    plot_ecdf_thresholds([whitept, nonwhitept],
                         ['White', 'Non-White'],
                         'Race',
                         dtime,
                         thresholds)

    plot_ecdf_thresholds([youngpt, midpt, oldpt],
                         ['18-27', '27-39', '>39'],
                         'Age',
                         dtime,
                         thresholds)

    # IAT/Broad rate calculations
    iat_broad_rate_race_csv(dfs_race,
                       dtime)

    iat_broad_rate_age_csv(dfs_age,
                       dtime)

    # Flowchart numbers
    general_make_counts(dfs_race,
                        'race',
                        dtime)

    general_make_counts(dfs_age,
                        'age_group',
                        dtime)

    # Plots of IAT/Broad for subgroups
    iat_broad_plot_race([whitept, nonwhitept],
                        dtime)

    iat_broad_plot_age([youngpt, midpt, oldpt],
                       dtime)
