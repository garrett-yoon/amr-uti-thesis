from data_process import *

print(add_labels(add_race_age(add_prescription(get_test_predictions()))).columns)
