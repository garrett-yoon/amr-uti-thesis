import pandas as pd
from functions import make_fig_fold
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
import shap
import json

print('Loading Data')
dataPath = Path('/Users/garrettyoon/Code/AMR-UTI/amr-uti-stm/data')

train_features_df = pd.read_csv(dataPath / 'train_uncomp_uti_features.csv')
test_features_df = pd.read_csv(dataPath / 'test_uncomp_uti_features.csv')

train_labels_df = pd.read_csv(dataPath / 'train_uncomp_resist_data.csv').drop(columns=['example_id'])
test_labels_df = pd.read_csv(dataPath / 'test_uncomp_resist_data.csv').drop(columns=['example_id'])

params_path = '/Users/garrettyoon/Code/AMR-UTI/amr-uti-stm/models/replication_hyperparameters/hyperparameters.json'
best_models_path = '/Users/garrettyoon/Code/AMR-UTI/amr-uti-stm/models/replication_hyperparameters/best_models.json'

with open(params_path) as f:
    hyperparams_by_abx = json.load(f)

with open(best_models_path) as f:
    best_models = json.load(f)

abx_list = ['NIT', 'SXT', 'CIP', 'LVX']

train_predictions_df = train_features_df[['example_id']].copy()
train_predictions_df['is_train'] = 1

test_predictions_df = test_features_df[['example_id']].copy()
test_predictions_df['is_train'] = 0
model_dict = {}

selector = VarianceThreshold()
selector.fit(train_features_df.drop(columns=['example_id']).values)

X_tr = selector.transform(train_features_df.drop(columns=['example_id']).values)
X_tr = pd.DataFrame(X_tr, index=train_features_df.index,
                    columns=train_features_df.drop(columns=['example_id']).columns)

X_te = selector.transform(test_features_df.drop(columns=['example_id']).values)
X_te = pd.DataFrame(X_te, index=test_features_df.index,
                    columns=test_features_df.drop(columns=['example_id']).columns)

for abx in abx_list:
    print(f"Starting model for {abx}")
    model = LogisticRegression()
    model.set_params(**hyperparams_by_abx[abx]['lr'])
    model.fit(X_tr, train_labels_df[abx].values)

    model_dict[abx] = model

    train_preds = model.predict_proba(X_tr)[:, 1]
    train_predictions_df[f'predicted_prob_{abx}'] = train_preds

    test_preds = model.predict_proba(X_te)[:, 1]
    test_predictions_df[f'predicted_prob_{abx}'] = test_preds

print("Making output folder")
dtime = make_fig_fold('_shap')

all_preds_df = pd.concat([train_predictions_df, test_predictions_df], axis = 0)
# all_preds_df.to_csv(f'{dtime}/all_preds.csv', index=False)

X100 = shap.utils.sample(X_te, 100)

explainer = shap.LinearExplainer(model_dict['NIT'], X_tr,
                                 feature_perturbation="interventional")

shap_values = explainer(X_te)


print(shap_values[:1, :])

shap.plots.waterfall(shap_values[0], show=False)

fig = plt.gcf()
fig.subplots_adjust(left=0.25)
fig.set_figheight(12)
fig.set_figwidth(24)
plt.xticks()

plt.savefig(f"{dtime}/waterfall_example.png")

shap.summary_plot(shap_values, X_te)

plt.savefig(f"{dtime}/summary_plot.png")