from functions import *
from data_process import *
from figures import *
import policy_learning_thresholding as plearn
from utils.evaluation_utils import get_iat_broad, get_iat_broad_bootstrapped

exp_res_path = pathlib.Path("../amr-uti-stm/experiments/experiment_results")
test_policy_df = pd.read_csv(exp_res_path / 'thresholding/thresholding_eval_test/results/test_policy_df.csv')
test_policy_df = add_race_age(test_policy_df)
test_policy_df_white = get_white_data(test_policy_df)
test_policy_df_nonwhite = get_nonwhite_data(test_policy_df)
policy_dfs = [test_policy_df_white, test_policy_df_nonwhite]

dtime = make_fig_fold()
DATA_PATH = '../amr-uti-stm/data'
PRED_PATH = '../amr-uti-stm/experiments/experiment_results/train_outcome_models/train_outcome_models_eval_test/results'
# Get training predictions
test_labels = pd.read_csv(f"{DATA_PATH}/test_uncomp_resist_data.csv")
train_labels = pd.read_csv(f"{DATA_PATH}/train_uncomp_resist_data.csv")
all_preds = pd.read_csv(f"{PRED_PATH}/test_predictions.csv")#.query('is_train == 0').drop(['is_train'], axis=1)
test_preds_actual = pd.merge(all_preds, test_labels, on='example_id')
train_preds_actual = pd.merge(all_preds, train_labels, on='example_id')

# Code to get thresholds
abx_list=['NIT', 'SXT', 'CIP', 'LVX']

val_outcomes_by_setting = pd.read_csv(exp_res_path / 'thresholding/thresholding_validation/results/val_stats_by_setting.csv')

constraint = 0.1

best_setting = val_outcomes_by_setting[
    val_outcomes_by_setting['broad_prop'] <= constraint
].sort_values(by='iat_prop').iloc[0]

curr_setting = dict(zip(abx_list, [{'vme': best_setting[abx]} for abx in abx_list]))

# Note that we choose the VME (i.e., false susceptibility rate) using the validation splits
# and then the "optimal threshold" corresponding to this VME is re-computed across the entire
# training set
thresholds = plearn.get_thresholds_dict(train_preds_actual, curr_setting, abx_list=abx_list)

plot_ecdf_thresholds(policy_dfs,
                     ['White', 'Non-White'],
                     'Race',
                     dtime,
                     thresholds
                     )