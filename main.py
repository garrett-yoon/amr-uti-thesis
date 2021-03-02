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

warnings.filterwarnings('ignore')

sns.set_style('darkgrid')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':


    # # predicted probabilities of resistance for multiple train/validataion splits
    # # across all antibiotic of interest using the optimal tuned hyperparameters 
    # val_preds_res = pd.read_csv(dataPath/'val_predictions.csv')
    # val_preds_res = val_preds_res.merge(df_uti_resist_lab.drop(columns=['is_train','uncomplicated']), 
    #                                     on='example_id', how='inner')\
    # .merge(df_prescript.drop(columns=['is_train']), on='example_id', how='inner')

    # # Predicited probabilities of resistance for all (training and test) specimens
    # # Using optimal tuned hyperparameters and reproduced pipeline 
    # test_predictions_opt = pd.read_csv(dataPath/'test_predictions_optimal.csv')
    # test_predictions_e2e = pd.read_csv(dataPath/'test_predictions_e2e.csv')

    # # Predicted probabilities, true resistance labels and observed prescription, 
    # # algorithm reccomendation, and final reccomednation (defaulting to observed
    # # prescription if algorithm defers)
    # test_policy_opt = pd.read_csv(dataPath/'test_policy_optimal.csv')
    # test_policy_e2e = pd.read_csv(dataPath/'test_policy_e2e.csv')

    # # Policy df with dropped predicted probabilities
    # test_rec_opt = test_policy_opt.copy()[['example_id', 'NIT', 'SXT', 'CIP', 'LVX', 'prescription', 'rec_final']]
    # test_rec_e2e = test_policy_e2e.copy()[['example_id', 'NIT', 'SXT', 'CIP', 'LVX', 'prescription', 'rec_final']]

    # # Age Group
    # all_uncomp_uti_features['race'] = all_uncomp_uti_features['demographics - is_white']
    # all_uncomp_uti_features.race.replace(to_replace=[0, 1], value=['non-white', 'white'], inplace=True)
    # all_uncomp_uti_features['age'] = all_uncomp_uti_features['demographics - age']

    # all_uncomp_uti_features['age_group'] = all_uncomp_uti_features.apply(age_group, axis=1)
    # all_uncomp_uti_features[['example_id', 'age_group']].head()

    # # Add Age and Race Data to test_predictions
    # id_race = all_uncomp_uti_features[['example_id', 'demographics - is_white', 'age_group', 'race']].rename(columns={"demographics - is_white": "white"})

    # val_preds_res_r = val_preds_res.merge(id_race, on='example_id',how='left')

    # test_predictions_opt_r = test_predictions_opt.merge(id_race, on='example_id',how='left')\
    # .merge(df_uti_resist_lab, on='example_id', how='left').drop(columns=['is_train_y', 'uncomplicated'])\
    # .merge(df_prescript, on='example_id', how='left').drop(columns=['is_train_x'])

    # test_predictions_e2e_r = test_predictions_e2e.merge(id_race, on='example_id',how='left')\
    # .merge(df_uti_resist_lab, on='example_id', how='left').drop(columns=['is_train_y', 'uncomplicated'])\
    # .merge(df_prescript, on='example_id', how='left').drop(columns=['is_train_x'])

    # test_policy_opt_r = test_policy_opt.merge(id_race, on='example_id',how='left')
    # test_policy_e2e_r = test_policy_e2e.merge(id_race, on='example_id',how='left')
    # test_rec_opt_r = test_rec_opt.merge(id_race, on='example_id',how='left')
    # test_rec_e2e_r = test_rec_e2e.merge(id_race, on='example_id',how='left')
    
    all_uncomp_uti_features, test_predictions_opt_r, test_policy_opt_r = dataprocess.get_features_df()


    test_predictions_a = test_predictions_opt_r.query('is_train == 0')
    test_predictions_w = test_predictions_opt_r.query('white == 1').query('is_train == 0')
    test_predictions_nw = test_predictions_opt_r.query('white == 0').query('is_train == 0')

    # Label train and validation dataframes with variables for whether
    # sample is non-susceptible to an antibiotic based on optimal thresholds
    # Create recomendations based on thresholds. Calculate IAT rate and Broad Use rate

    # Thresholds
    thresholds = {"NIT": 0.129, 
                "SXT": 0.18, 
                "CIP": 0.258, 
                "LVX": 0.239} 



    # Recommendations for thresholds
    allpt, whitept, nonwhitept = create_rec_df_race(test_predictions_opt_r,
                                                    thresholds)

    _, youngpt, midpt, oldpt = create_rec_df_age(test_predictions_opt_r,
                                                    thresholds)

    dfs_race = [allpt, whitept, nonwhitept]
    dfs_age = [allpt, youngpt, midpt, oldpt]

    # Make the directory for figure/csv outputs
    dtime = make_fig_fold()

    thresholds_df =dataprocess.pd.DataFrame(thresholds, index=['Thresholds'])
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
    iat_broad_rate_csv(dfs_race,
                    dtime)

    iat_broad_rate_age(dfs_age,
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

