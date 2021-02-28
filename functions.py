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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




# Create age groups using pandas apply
def age_group(x):
    q1 = 27.0
    q2 = 40.0
    
    if x.age <= q1:
        return 0
    elif x.age <= q2:
        return 1
    else:
        return 2

# Creates binary labels for whether the sample passes the threshold for each abx
def create_binary_suscept(df, thresholds):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    new_df = df
    for abx in abxs:
        var = 'predicted_prob_' + abx
        new = 'nonsusceptible_' + abx
        t_val = thresholds[abx]
        new_df[new] = (new_df[var] > t_val).astype(int)
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

# Create ECDF Plots
def plot_ecdf_thresholds(data_list, 
                         labels, 
                         split_s, 
                         time,
                         thresholds):
    abxs = ['NIT', 'SXT', 'CIP', 'LVX']
    data_list = data_list.copy()
    fig, ax = plt.subplots(2,2, figsize=(20, 10))
    axes = [ax for ax in fig.axes]
    tup = zip(abxs, axes)
    colors = ['blue', 'darkorange', 'green']
    patches = []

    for abx, ax in tup:
        
        # Create ECDF for the antibiotic for each group
        lab = 'predicted_prob_' + abx
        for df, c, label in zip(data_list, colors[:len(data_list)], labels):
            if df.shape[0] == 0:
                continue
            ecdf = ECDF(df[lab])
            ax.plot(ecdf.x, ecdf.y, color=c)

        # Make plot
        ax.set_title(abx)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('% of Observations')
        ax.axvline(thresholds[abx], linestyle='--', color='black')
        ax.text(x = thresholds[abx] + 0.1, 
            y = 0.05, 
            s = 'Threshold:' + str(round(thresholds[abx], 3)),
            size = 'large')
        ax.grid(b=True)

        # Subset dataframe to not include the prev antibiotic recs
        for i, df in enumerate(data_list):
            data_list[i] = df[df.rec_final != abx]
    
    for c, label in zip(colors, labels):
        p = mpatches.Patch(color=c, label=label)
        patches.append(p)

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.90)
    plt.legend(title=split_s, handles=patches)
    plt.suptitle('Cumulative Distributions With Optimal Thresholds Conditional on ' + split_s
                        , size='20')
    plt.savefig(f"{time}/{time}" + '_ECDF_' +f'{split_s}' 
                + '.png', dpi=300)


# Splits by race, and makes recommendations + labels IAT/Broad
def create_rec_df_race(df, thresholds):
    test_predictions_a = df.query('is_train == 0')
    test_predictions_w = df.query('white == 1').query('is_train == 0')
    test_predictions_nw = df.query('white == 0').query('is_train == 0')

    for df in [test_predictions_a, test_predictions_w, test_predictions_nw]:
        df = create_binary_suscept(df, thresholds)
        df['rec'] = df.apply(create_recomendation, axis = 1) 
        df['rec_final'] = df.apply(create_recomendation_final, axis = 1)
        df['iat'] = df.apply(iat, axis = 1)
        df['broad_abx'] = df.apply(broad_abx, axis = 1)
    
    return test_predictions_a, test_predictions_w, test_predictions_nw

# Splits by age, and makes recs + labels IAT/Broad
def create_rec_df_age(df, thresholds):
    
    # By age group analysis

    test_predictions_a = df.query('is_train == 0')
    test_predictions_young = df.query('age_group == 0').query('is_train == 0')
    test_predictions_mid = df.query('age_group == 1').query('is_train == 0')
    test_predictions_old = df.query('age_group == 2').query('is_train == 0')

    # Label train and validation dataframes with variables for whether
    # sample is non-susceptible to an antibiotic based on optimal thresholds

    for df in [test_predictions_a, test_predictions_young, test_predictions_mid, test_predictions_old]:
        df = create_binary_suscept(df, thresholds)
        df['rec'] = df.apply(create_recomendation, axis = 1) 
        df['rec_final'] = df.apply(create_recomendation_final, axis = 1)
        df['iat'] = df.apply(iat, axis = 1)
        df['broad_abx'] = df.apply(broad_abx, axis = 1)
    
    return test_predictions_a, test_predictions_young, test_predictions_mid, test_predictions_old


# Calculate IAT and Broad Rate by Race
def iat_broad_rate_csv(dfs, time): 
    outcomes = pd.DataFrame()
    for df, g in list(zip(dfs, ['All Pts','White','Non-White'])):
        outcomes.loc[g, 'IAT_rate'] = iat_rate(df)
        outcomes.loc[g, 'Broad_rate'] = broad_abx_rate(df)
        
    outcomes.to_csv(f"{time}/{time}" + '_iat_broad_rates_race.csv')
    
# Calculate IAT and Broad Rate by Age
def iat_broad_rate_age(dfs, time): 
    outcomes = pd.DataFrame()
    for df, g in list(zip(dfs, ['All Pts','Young','Mid', 'Old'])):
        outcomes.loc[g, 'IAT_rate'] = iat_rate(df)
        outcomes.loc[g, 'Broad_rate'] = broad_abx_rate(df)
        
    outcomes.to_csv(f"{time}/{time}" + '_iat_broad_rates_age.csv')
    
# Make counts for flowchart
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

# Plots IAT vs Broad conditional on Race
def iat_broad_plot_race(dfs, time):

    white_pt = get_iat_broad(dfs[0],'rec_final')
    n_white_pt = get_iat_broad(dfs[1],'rec_final')
    plt.subplots(figsize=(10,10))
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.plot([white_pt[1], n_white_pt[1]], [white_pt[0], n_white_pt[0]], marker = 'o', 
            zorder=1, color = 'black', alpha=0.8)
    
    paper_pt = [0.11, 0.098]
    plt.scatter(paper_pt[0], paper_pt[1], color='grey', s=200,zorder=2)
    plt.text(white_pt[1] + 0.005, white_pt[0] + 0.01, 'White', color='blue')
    #plt.scatter(all_pt[1], all_pt[0], color='grey', s=200,zorder=2)
    
    plt.scatter(white_pt[1], white_pt[0], color='blue', s=200, zorder=2)
    plt.scatter(n_white_pt[1], n_white_pt[0], color='orange', s=200, zorder=2)
    plt.text(white_pt[1] + 0.005, white_pt[0] + 0.01, 'White', color='blue')
    plt.text(n_white_pt[1] + 0.005, n_white_pt[0] + 0.01, 'Non-White', color='orange')
    plt.text(paper_pt[0] + 0.005, paper_pt[1] + 0.01, 'Published Result', color='grey')
    plt.xlabel('% Broad Spectrum Antibiotic Use',size=16)
    plt.xticks(size=14)
    plt.ylabel('% Inappropiate Antibiotic Therapy Use', size=16)
    plt.yticks(size=14)
    plt.title("IAT vs % Broad Spectrum Use Conditional on Racial Subgroup", 
              size=20,
              pad = 20)
    plt.savefig(f"{time}/{time}" + "_iat_broad_race.png", dpi=300)

# Plots IAT and Broad conditional on Age
def iat_broad_plot_age(dfs, time):

    pts = []
    for g in dfs:
        get_iat_broad(g, 'rec_final')[1]
        pts.append((get_iat_broad(g, 'rec_final')[1], get_iat_broad(g, 'rec_final')[0]))
    paper_pt = [0.11, 0.098]
    plt.subplots(figsize=(10,10))
    for i, k in enumerate(list(zip(['Group 0', 'Group 1', 'Group 2'], ['blue','darkorange', 'green']))):
        plt.scatter(pts[i][0], pts[i][1], marker='o',color=k[1], s=200)
        plt.text(pts[i][0] + 0.005, pts[i][1] + 0.01, s=k[0], color= k[1])
    plt.text(paper_pt[0] -0.04, paper_pt[1]+0.005, 'Published Result', color='grey')
    plt.scatter(paper_pt[0], paper_pt[1], color='grey', s=200,zorder=2)
    plt.xlim(0, 0.3)
    plt.ylim(0, 0.2)
    plt.xlabel('% Broad Spectrum Antibiotic Use',size='large')
    plt.ylabel('% Inappropiate Antibiotic Therapy Use', size='large')
    plt.title("IAT vs % Broad Spectrum Use Conditional on Age Subgroup", size='x-large')
    plt.legend(labels=['18-27', '27-39', '>39'])
    plt.savefig(f"{time}/{time}" + "_iat_broad_age.png", dpi=300)