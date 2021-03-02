import unittest
from data_process import *


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

