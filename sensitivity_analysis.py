from thresholding import *
from data_process import *
from figures import *


def thresholding_analysis(dtime):

    white_stats_df = load_data('replicated_stats_df_white.csv')
    nonwhite_stats_df = load_data('replicated_stats_df_nonwhite.csv')
    train_preds_actual = load_data('train_preds_actual.csv')

    w_best_threshold, w_best_setting = get_best_setting_threshold(white_stats_df)
    nw_best_threshold, nw_best_setting = get_best_setting_threshold(nonwhite_stats_df)
    w_iat_prop, w_broad_prop = w_best_setting[['iat_prop', 'broad_prop']]
    nw_iat_prop, nw_broad_prop = nw_best_setting[['iat_prop', 'broad_prop']]

    plot_thresholds_stats_by_race(white_stats_df, nonwhite_stats_df, dtime)

