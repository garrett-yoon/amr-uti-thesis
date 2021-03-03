import unittest

from pandas.testing import assert_frame_equal, assert_series_equal

from data_process import *
from functions import *


class TestDataProcess(unittest.TestCase):

    def test_load_data(self):
        self.assertEqual(get_features_df().shape[0], 15806)
        self.assertEqual(get_val_preds().shape[0], 11865 * 20)
        self.assertEqual(get_test_predictions().shape[0], 15806)
        self.assertEqual(get_test_policy_df().shape[0], 3941)
        self.assertEqual(add_race_age(get_test_predictions()).shape[0], 15806)
        self.assertEqual(get_train_test(get_features_df())[0].shape[0], 11865)
        self.assertEqual(get_train_test(get_features_df())[1].shape[0], 3941)

    def test_add_prescriptions(self):
        self.assertEqual(add_prescription(get_test_predictions()).shape[0], 15806)

    def test_white_nonwhite(self):
        self.assertEqual(get_white_data(add_race_age(get_test_predictions())).shape[0], 10054)
        self.assertEqual(get_nonwhite_data(add_race_age(get_test_predictions())).shape[0], 5752)


class TestFunctions(unittest.TestCase):

    pd.set_option('display.max_columns', 500)

    def test_create_binary_nonsuscept(self):
        df = pd.DataFrame({'predicted_prob_NIT': [0.2],
                           'predicted_prob_SXT': [0.05],
                           'predicted_prob_CIP': [0.2],
                           'predicted_prob_LVX': [0.05]})
        thresholds = {'NIT': 0.1, 'SXT': 0.1, 'CIP': 0.1, 'LVX': 0.1}
        temp = create_binary_nonsuscept(df, thresholds)
        labels = pd.DataFrame({'nonsusceptible_NIT': [1],
                               'nonsusceptible_SXT': [0],
                               'nonsusceptible_CIP': [1],
                               'nonsusceptible_LVX': [0]})
        expected = pd.concat([df, labels], axis=1)
        assert_frame_equal(temp, expected)

    def test_create_recommendation(self):
        labels = pd.DataFrame({'nonsusceptible_NIT': [1],
                               'nonsusceptible_SXT': [0],
                               'nonsusceptible_CIP': [1],
                               'nonsusceptible_LVX': [0]})
        assert_series_equal(labels.apply(create_recomendation, axis=1), pd.Series(['SXT']))

    def test_create_recommendation_final(self):
        labels = pd.DataFrame({'nonsusceptible_NIT': [1],
                               'nonsusceptible_SXT': [1],
                               'nonsusceptible_CIP': [1],
                               'nonsusceptible_LVX': [1],
                               'prescription': 'SXT'})
        assert_series_equal(labels.apply(create_recomendation_final, axis=1), pd.Series(['SXT']))
