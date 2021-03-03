import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import roc_curve
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
import itertools
import os
import pathlib
from datetime import datetime

from collections import defaultdict

import matplotlib.pyplot as plt


# Creates binary labels for whether the sample passes the threshold for each abx
def create_binary_nonsuscept(df, thresholds):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    new_df = df.copy()
    for abx in abxs:
        var = 'predicted_prob_' + abx
        new = 'nonsusceptible_' + abx
        t_val = thresholds[abx]
        new_df[new] = (new_df[var] > t_val).astype('int64')
    return new_df


# Recommends the most narrow antibiotic pt is susceptible to
def create_recomendation(row):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    for abx in abxs:
        lab = 'nonsusceptible_' + abx
        if row[lab] == 0:
            return abx
    return 'defer'


# Recommends the most narrow antibiotic pt is susceptible to
# If defer, recommends actual prescription given
def create_recomendation_final(row):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    for abx in abxs:
        lab = 'nonsusceptible_' + abx
        if row[lab] == 0:
            return abx
    return row.prescription


# Returns 1 if patient is resistant to rec abx
def iat(row):
    abx = row.rec_final
    return int(row[abx] == 1)


# Calculates the IAT Rate
def iat_rate(df):
    if df.shape[0] == 0:
        return 0
    return str(round(sum(df['iat'])/(df.shape[0]), 5)) 


# Returns 1 if patient is given CIP or LVX
def broad_abx(row):
    return int(row.rec_final in ['CIP', 'LVX'])


# Calculates the 2nd Line Rate
def broad_abx_rate(df):
    if df.shape[0] == 0:
        return 0
    return str(round(sum(df['broad_abx'])/(df.shape[0]), 5)) 


# Returns a tuple of the IAT and Broad rates
def get_iat_broad(policy_df, col_name):
    iat = policy_df.apply(lambda x: x[f'{x[col_name]}']==1, axis=1).mean()
    broad = policy_df.apply(lambda x: x[col_name] in ['CIP', 'LVX'], axis=1).mean() 
    return iat, broad


def make_fig_fold():
    t = datetime.now().strftime("%d-%m-%Y_%H-%M-%S") 
    os.mkdir(t)
    return t 


# Splits by race, and makes recommendations + labels IAT/Broad
def create_rec_dfs(dfs, thresholds):

    for df in dfs:
        df = create_binary_suscept(df, thresholds)
        df['rec'] = df.apply(create_recomendation, axis = 1)
        df['rec_final'] = df.apply(create_recomendation_final, axis = 1)
        df['iat'] = df.apply(iat, axis = 1)
        df['broad_abx'] = df.apply(broad_abx, axis = 1)

    return dfs


# Calculate IAT and Broad Rate by Race and store as csv
def iat_broad_rate_race_csv(dfs, time):
    outcomes = pd.DataFrame()
    for df, g in list(zip(dfs, ['All Pts','White','Non-White'])):
        outcomes.loc[g, 'IAT_rate'] = iat_rate(df)
        outcomes.loc[g, 'Broad_rate'] = broad_abx_rate(df)
        
    outcomes.to_csv(f"{time}/{time}" + '_iat_broad_rates_race.csv')


# Calculate IAT and Broad Rate by Age and store as csv
def iat_broad_rate_age_csv(dfs, time):
    outcomes = pd.DataFrame()
    for df, g in list(zip(dfs, ['All Pts','Young','Mid', 'Old'])):
        outcomes.loc[g, 'IAT_rate'] = iat_rate(df)
        outcomes.loc[g, 'Broad_rate'] = broad_abx_rate(df)
        
    outcomes.to_csv(f"{time}/{time}" + '_iat_broad_rates_age.csv')


# Make counts for flowchart for specfied subgroups (race vs age_group)
def general_make_counts(data, column, time):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    counts = {}
    for group in data:
        temp = group.copy()
        name = group[column].iloc[0]
        counts[name] = {}
        counts[name]['TOTAL'] = group.shape[0]
        curr = counts[name]['TOTAL'] 
        for abx in abxs:
            temp = group[group['rec'] == abx]
            iat_r = round(sum(temp['iat']/temp['iat'].shape),4)
            yes_ct = temp.shape[0]
            if yes_ct == 0:
                counts[name][abx + "_YES"] = yes_ct
                counts[name][abx + "_YES%"] = 0
            else:  
                counts[name][abx + "_YES"] = int(yes_ct)
                counts[name][abx + "_YES%"] = round(yes_ct/curr,4)
            counts[name][abx + "_NO"] = int(curr - yes_ct)
            counts[name][abx + "_NO%"] = round((curr -yes_ct)/curr,4)
            counts[name][abx + "_APP%"] =  round((1-iat_r)*100, 4)
            curr = curr - yes_ct
        counts[name]['DEFER'] = curr
    out = pd.DataFrame(counts)   
    out.to_csv(f"{time}/{time}" + f'_flowchart_numbers_{column}.csv')
    return out
    
