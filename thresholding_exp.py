from thresholding import *
from data_process import *
from functions import make_fig_fold
import warnings
warnings.filterwarnings('ignore')

'''This file will create thresholding by race given for a specific validation predications dataframe'''

# CHANGE TO TRUE IF SPLIT BY RACE
split_by_race = False

# Get train/validation predictions
val_preds = get_val_preds()
val_preds = add_race_age(val_preds)
white_val_preds = get_white_data(val_preds)
nonwhite_val_preds = get_nonwhite_data(val_preds)

# Create output folder
date_time = make_fig_fold('_thresholding_exp')

# setting_combos = create_setting_combos([0.001, 0.015, 0.1, 0.2, 0.3,
#                                         0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                                        abx_list=abx_list)

setting_combos = create_setting_combos([0.3, 0.6],
                                       abx_list=abx_list)

stat_columns = [
    'iat_prop', 'broad_prop', 'iat_prop_decision', 'broad_prop_decision',
    'iat_diff_mean', 'iat_diff_std', 'broad_diff_mean', 'broad_diff_std',
    'defer_rate', 'NIT_thresh', 'SXT_thresh', 'CIP_thresh', 'LVX_thresh']

if split_by_race:
    dfs = {'white': white_val_preds, 'non-white': nonwhite_val_preds}.items()
else:
    dfs = {'all': val_preds}


# Loop through white and non-white dataframes
for lab, df in dfs.items():

    stats_by_setting = []

    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']

    current_df = df
    setting_combos = list(setting_combos)
    if not False:

        for i, setting in enumerate(setting_combos):

            curr_setting = dict(zip(abx_list, setting))

            # Skip combination when VME for CIP and LVX are same
            if curr_setting['CIP']['vme'] != curr_setting['LVX']['vme']:
                continue
            print(f'Working on combination {i} / {len(setting_combos)}')
            stats_for_curr_setting = defaultdict(list)

            num_splits = 20

            for split in range(num_splits):
                # Get the current split of data
                preds_for_split = current_df[current_df['split_ct'] == split]

                # Get train/val predictions
                train_preds = preds_for_split[preds_for_split['is_train'] == 1].copy()
                val_preds = preds_for_split[preds_for_split['is_train'] == 0].copy()

                # Using train/val predictions, get the stats
                stats_dict_for_split, thresh_for_split = get_stats_for_train_val_preds(
                    train_preds, val_preds, curr_setting, contra_dict=None,
                    abx_list=abx_list,
                )

                # Get the the rates of IAT and 2nd line of the administered prescriptions by physicians
                doc_iat, doc_broad = get_iat_broad(val_preds, col_name='prescription')

                # Calculate the difference of IAt and 2nd line between algorithm and physicians
                stats_dict_for_split['iat_diff'] = (stats_dict_for_split['iat_prop'] - doc_iat) * len(val_preds)
                stats_dict_for_split['broad_diff'] = (stats_dict_for_split['broad_prop'] - doc_broad) * len(val_preds)

                # Append the statistics to a dictionary for a given split
                for stat in stats_dict_for_split.keys():
                    stats_for_curr_setting[stat].append(stats_dict_for_split[stat])

            #
            compiled_stats_for_setting = [curr_setting]

            # Calculate the mean statistics across 20 splits
            for stat in stat_columns:
                if stat.endswith('_mean'):
                    stat_name = stat[:stat.index('_mean')]
                    compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat_name]))

                elif stat.endswith('_std'):
                    stat_name = stat[:stat.index('_std')]
                    compiled_stats_for_setting.append(np.std(stats_for_curr_setting[stat_name]))

                else:
                    compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat]))

            # Append statistics for a given setting to output dictionary
            stats_by_setting.append(compiled_stats_for_setting)

        # Create dictionary to dataframe
        stats_df = convert_dict_to_df(stats_by_setting, stat_columns=stat_columns, abx_list=abx_list)

        print(f'Done with {lab} group stats')

        # Output dictionary for statistics
        stats_df.to_csv(pathlib.Path(date_time) / f'replicated_stats_df_{lab}.csv', index=False)
