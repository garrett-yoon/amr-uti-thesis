from data_process import *
from collections import defaultdict
import itertools
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np

# Using Figure 3 data, find the threshold value given the FNR
abx_list = ['NIT', 'SXT', 'CIP', 'LVX']
val_stats_by_setting = load_data("val_stats_by_setting.csv")
fnrs = val_stats_by_setting[['NIT', 'SXT', 'CIP', 'LVX']]


def get_policy_defer(row, thresholds, abx_list=None):
    if abx_list is None:
        abx_list = ['NIT', 'SXT', 'CIP', 'LVX']
    for abx in abx_list:
        if row[f'predicted_prob_{abx}'] < thresholds[abx]:
            return abx

    return "defer"


def get_policy_for_preds(preds_df, thresholds, contra_dict=None,
                         abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    policy_df = preds_df.copy()

    if contra_dict is not None:
        policy_df['rec'] = policy_df.apply(
            lambda x: get_policy_with_contraindications(x, thresholds, contra_dict,
                                                        abx_list=abx_list), axis=1
        )

    else:
        policy_df['rec'] = policy_df.apply(
            lambda x: get_policy_defer(x, thresholds, abx_list=abx_list), axis=1
        )

    # Fill in deferral with actual antibiotic name
    policy_df['rec_final'] = policy_df.apply(
        lambda x: x['prescription'] if x['rec'] == 'defer' else x['rec'], axis=1
    )

    return policy_df


def get_iat_broad(policy_df, col_name):
    iat = policy_df.apply(lambda x: x[f'{x[col_name]}'] == 1, axis=1).mean()
    broad = policy_df.apply(lambda x: x[col_name] in ['CIP', 'LVX'], axis=1).mean()

    return iat, broad


def create_setting_combos(fnr_vals, abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    label_settings = defaultdict(list)

    for abx in abx_list:
        label_settings[abx].extend([{'vme': vme} for vme in fnr_vals])

    settings = [label_settings[abx] for abx in abx_list]
    setting_combos = itertools.product(*settings)

    return setting_combos


def convert_dict_to_df(stats_by_setting, stat_columns,
                       abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    data = []

    for setting in stats_by_setting:
        data_for_setting = [setting[0][abx]['vme'] for abx in abx_list] + list(setting[1:])
        data.append(data_for_setting)

    return pd.DataFrame(data, columns=abx_list + stat_columns)


def get_threshold(is_resistant, resist_probs, fnr):
    desired_tpr = 1 - fnr
    fprs, tprs, thresholds = roc_curve(is_resistant,
                                       resist_probs)

    diffs = [abs(t - desired_tpr) for t in tprs]
    i = diffs.index(min(diffs))
    return thresholds[i], fprs[i], tprs[i]


def get_thresholds_dict(preds_df, fnr_setting, abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    thresholds = {}

    for abx in abx_list:
        threshold, _, _ = get_threshold(preds_df[abx].values,
                                        preds_df[f'predicted_prob_{abx}'].values,
                                        fnr_setting[abx]['vme'])
        thresholds[abx] = threshold
    return thresholds


def get_stats_for_train_val_preds(train_preds, val_preds, curr_setting, contra_dict=None,
                                  bootstrap=False, save_policy=False,
                                  abx_list=['NIT', 'SXT', 'CIP', 'LVX']):
    thresholds = get_thresholds_dict(train_preds, curr_setting, abx_list=abx_list)
    val_policy_df = get_policy_for_preds(val_preds, thresholds,
                                         contra_dict=contra_dict, abx_list=abx_list)

    decision_cohort = val_policy_df[val_policy_df['rec'] != 'defer']

    if bootstrap:
        val_iat, val_broad = get_iat_broad_bootstrapped(val_policy_df, col_name='rec_final')
        val_iat_decision, val_broad_decision = get_iat_broad_bootstrapped(decision_cohort, col_name='rec_final')
    else:
        val_iat, val_broad = get_iat_broad(val_policy_df, col_name='rec_final')
        val_iat_decision, val_broad_decision = get_iat_broad(decision_cohort, col_name='rec_final')

    res = {'iat_prop': val_iat,
           'broad_prop': val_broad,
           'iat_prop_decision': val_iat_decision,
           'broad_prop_decision': val_broad_decision,
           'defer_rate': 1 - (len(decision_cohort) / len(val_policy_df))}

    if save_policy:
        return res, val_policy_df, thresholds
    else:
        return res, thresholds


def get_policy_with_contraindications(row, thresholds, contra_dict, abx_list=['NIT', 'SXT', 'CIP', 'LVX']):

    for abx in abx_list:
        if row[f'predicted_prob_{abx}'] < thresholds[abx] and row['example_id'] not in contra_dict[abx]:
            return abx

    return "defer"


def get_iat_broad_bootstrapped(policy_df, col_name, num_samples=20):
    iats, broads = [], []

    for i in range(num_samples):
        policy_df_sampled = policy_df.sample(n=len(policy_df), replace=True,
                                             random_state=10 + i)

        iat, broad = get_iat_broad(policy_df_sampled, col_name=col_name)
        iats.append(iat)
        broads.append(broad)

    return np.mean(iats), np.mean(broads)


def get_best_setting_threshold(df):
    constraint = 0.1

    train_preds_actual = load_data('train_preds_actual.csv')

    best_setting = df[
        df['broad_prop'] <= constraint
        ].sort_values(by='iat_prop').iloc[0]

    curr_setting = dict(zip(abx_list, [{'vme': best_setting[abx]} for abx in abx_list]))

    # Note that we choose the VME (i.e., false susceptibility rate) using the validation splits
    # and then the "optimal threshold" corresponding to this VME is re-computed across the entire
    # training set
    thresholds = get_thresholds_dict(train_preds_actual, curr_setting, abx_list=abx_list)

    return thresholds, best_setting

