from thresholding import *
from data_process import *
from functions import make_fig_fold
import warnings
warnings.filterwarnings('ignore')

val_preds = get_val_preds()
val_preds = add_race_age(val_preds)
white_val_preds = get_white_data(val_preds)
nonwhite_val_preds = get_nonwhite_data(val_preds)

date_time = make_fig_fold('_thresholding_stats')

# setting_combos = create_setting_combos([0.001, 0.015, 0.1, 0.2, 0.3,
#                                         0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#                                        abx_list=abx_list)

setting_combos = create_setting_combos([0.3, 0.6],
                                       abx_list=abx_list)

stat_columns = [
    'iat_prop', 'broad_prop', 'iat_prop_decision', 'broad_prop_decision',
    'iat_diff_mean', 'iat_diff_std', 'broad_diff_mean', 'broad_diff_std',
    'defer_rate', 'NIT_thresh', 'SXT_thresh', 'CIP_thresh', 'LVX_thresh']

for lab, df in {'white': white_val_preds, 'non-white': nonwhite_val_preds}.items():

    stats_by_setting = []

    abx_list = ['NIT', 'SXT', 'CIP', 'LVX']

    current_df = df
    setting_combos = list(setting_combos)
    if not False:

        for i, setting in enumerate(setting_combos):

            curr_setting = dict(zip(abx_list, setting))
            if curr_setting['CIP']['vme'] != curr_setting['LVX']['vme']:
                continue
            print(f'Working on combination {i} / {len(setting_combos)}')
            stats_for_curr_setting = defaultdict(list)

            num_splits = 20

            for split in range(num_splits):

                preds_for_split = current_df[current_df['split_ct'] == split]

                # Get train/val predictions
                train_preds = preds_for_split[preds_for_split['is_train'] == 1].copy()
                val_preds = preds_for_split[preds_for_split['is_train'] == 0].copy()

                stats_dict_for_split, thresh_for_split = get_stats_for_train_val_preds(
                    train_preds, val_preds, curr_setting, contra_dict=None,
                    abx_list=abx_list,
                )

                doc_iat, doc_broad = get_iat_broad(val_preds, col_name='prescription')

                stats_dict_for_split['iat_diff'] = (stats_dict_for_split['iat_prop'] - doc_iat) * len(val_preds)
                stats_dict_for_split['broad_diff'] = (stats_dict_for_split['broad_prop'] - doc_broad) * len(val_preds)

                for stat in stats_dict_for_split.keys():
                    stats_for_curr_setting[stat].append(stats_dict_for_split[stat])

            compiled_stats_for_setting = [curr_setting]

            for stat in stat_columns:
                if stat.endswith('_mean'):
                    stat_name = stat[:stat.index('_mean')]
                    compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat_name]))

                elif stat.endswith('_std'):
                    stat_name = stat[:stat.index('_std')]
                    compiled_stats_for_setting.append(np.std(stats_for_curr_setting[stat_name]))

                else:
                    compiled_stats_for_setting.append(np.mean(stats_for_curr_setting[stat]))

            stats_by_setting.append(compiled_stats_for_setting)

        stats_df = convert_dict_to_df(stats_by_setting, stat_columns=stat_columns, abx_list=abx_list)
        print(f'Done with {lab} group stats')

        stats_df.to_csv(pathlib.Path(date_time) / f'replicated_stats_df_{lab}.csv', index=False)
