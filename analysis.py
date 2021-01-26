# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# %%
df_prescript  = pd.read_csv('data/all_prescriptions.csv')
df_uti_features = pd.read_csv('data/all_uti_features.csv')
df_uti_resist_lab = pd.read_csv('data/all_uti_resist_labels.csv')
data_dict = pd.read_csv('data/data_dictionary.csv')
test_uncomp_resist_data =  pd.read_csv('data/test_uncomp_resist_data.csv')
test_uncomp_uti_features =  pd.read_csv('data/test_uncomp_uti_features.csv')
train_uncomp_resist_data =  pd.read_csv('data/train_uncomp_resist_data.csv')
train_uncomp_uti_features =  pd.read_csv('data/train_uncomp_uti_features.csv')


# %%
all_uncomp_uti_features = test_uncomp_uti_features.append(train_uncomp_uti_features)
all_uncomp_uti_features.shape[0]


# %%
print('The size of the uncomp train set', test_uncomp_resist_data.shape[0])
print('The size of the uncomp train set', train_uncomp_resist_data.shape[0])


# %%
# Find Feature Lists

starting = 'infection_site'
df_prescript.head()
features = [s for s in list(df_uti_features.isna().sum().index) if s.startswith(starting)]
print('\n'.join(features))
print('Number of Features:', len(features))


# %%
# Training set demographic distribution
count = train_uncomp_uti_features["demographics - is_white"].value_counts()
sns.set_style('whitegrid')
g = sns.barplot(x = count.index, y = count)
g.set(ylabel='Number of Observations')
_ = g.set(xticklabels=['Non-White', 'White'])


# %%
sns.violinplot(x = 'demographics - is_white', y = 'demographics - age', data = train_uncomp_uti_features)


# %%
# Training set demographic distribution
count = all_uncomp_uti_features["demographics - is_white"].value_counts()
sns.set_style('whitegrid')
g = sns.barplot(x = count.index, y = count)
g.set(ylabel='Number of Observations')
_ = g.set(xticklabels=['Non-White', 'White'])


# %%
sns.violinplot(x = 'demographics - is_white', y = 'demographics - age', data = all_uncomp_uti_features)


# %%
pie = pd.DataFrame({'count': [3, 4 , 144, 224, 81, 52, 36, 155, 74, 4, 6, 5]},
                  index=['demographics', 'location_type', 'prior_abx_rest','prior_abx_exp_med', 'prior_abx_exp_class','prior_abx_exp_subclass', 'prior_infect_org', 'comorbidities','colonization_pressure', 'skilled_nursing', 'other_infection', 'prior_procedures' ])
plot = pie.plot.pie(y='count', figsize=(5, 5))
plot.set_title("Breakdown of Features Numbers")
plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')


# %%
_  =  sns.heatmap(pie, annot=True, fmt="d",  cmap="YlGnBu", square = True)
# fix for mpl bug that cuts off top/bottom of seaborn viz
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() # ta-da!


# %%
df_uti_resist_lab.head()


# %%
# Study only looked at uncomplicated UTI

df_uti_resist_lab.groupby(['is_train', 'uncomplicated']).size()


# %%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data_dict.head()


# %%
all_proba = pd.read_csv("../amr-uti-stm/experiments/experiment_results/train_outcome_models/train_outcome_models_eval_test_rep/results/test_predictions.csv")
test_proba = all_proba.query("is_train == 0")
train_proba = all_proba.query("is_train == 1")
print(test_proba.shape)
print(train_proba.shape)

# %%

is_white = (all_uncomp_uti_features['demographics - is_white'] == 1)
is_nonwhite = (all_uncomp_uti_features['demographics - is_white'] == 0)
print('White:', sum(is_white),'\nNon-white:', sum(is_nonwhite))

# %%
white_proba = all_proba[is_white] 
nonwhite_proba = all_proba[is_nonwhite] 

# %%
