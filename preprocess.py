import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import sys
from util import encode_dataset, encode_split_dataset, get_unique_df_values, load_obj, map_columns, path, prepare_dataset, save_obj, split

sys.path.insert(0, path)

def preprocess(dataset, explainer):

    # Dataset preparation
    data =  class_name = None

    encoded_data = None

    if (dataset == 'texas'): #
        class_name = 'PRINC_SURG_PROC_CODE'
        dtypes = load_obj('data/' + dataset + '/dtypes')
        data = pd.read_csv('data/' + dataset + '/' + dataset +'_mapped.csv', dtype = dtypes)

        print("Splitting ...")
        bb_train, bb_val, sh_train, sh_val, r2E, test = split(data, class_name)

        bb_train.to_csv('data/' + dataset + '/baseline_split/bb_train_mapped.csv', index=False)
        print("bb_train saved")
        bb_val.to_csv('data/' + dataset + '/baseline_split/bb_val_mapped.csv', index=False)
        print("bb_val saved")
        sh_train.to_csv('data/' + dataset + '/baseline_split/sh_train_mapped.csv', index=False)
        print("sh_train saved")
        sh_val.to_csv('data/' + dataset + '/baseline_split/sh_val_mapped.csv', index=False)
        print("sh_val saved")
        r2E.to_csv('data/' + dataset + '/baseline_split/r2E_mapped.csv', index=False)
        print("r2E saved")
        test.to_csv('data/' + dataset + '/baseline_split/test_mapped.csv', index=False)

        return True
    else:
        data, class_name = prepare_dataset(dataset, explainer)

        data.to_csv("prepared_adult.csv")
        # Mapping
        mapped_data = map_columns(data, class_name)
        mapped_data.to_csv("mapped_adult.csv")

        # Encoding
        encoded_data, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = encode_dataset(data, class_name)
        encoded_data.to_csv("encoded_adult.csv")

    # Splitting both datasets
    bb_train, bb_val, sh_train, sh_val, r2E, test = split(data, class_name)
    bb_train_m, bb_val_m, sh_train_m, sh_val_m, r2E_m, test_m = split(mapped_data, class_name)
    bb_train_e, bb_val_e, sh_train_e, sh_val_e, r2E_e, test_e = split(encoded_data, class_name)

    # Writing datasets
    if(len(bb_train) + len(bb_val) + len(sh_train) + len(sh_val) + len(r2E) + len(test) == len(data)
    and len(bb_train_e) + len(bb_val_e) + len(sh_train_e) + len(sh_val_e) + len(r2E_e) + len(test_e) == len(encoded_data)):
        print('Dataset: ' + dataset)
        bb_train.to_csv('data/' + dataset + '/baseline_split/bb_train.csv', index=False)
        bb_train_m.to_csv('data/' + dataset + '/baseline_split/bb_train_mapped.csv', index=False)
        bb_train_e.to_csv('data/' + dataset + '/baseline_split/bb_train_e.csv', index=False)
        print("bb_train saved")
        bb_val.to_csv('data/' + dataset + '/baseline_split/bb_val.csv', index=False)
        bb_val_m.to_csv('data/' + dataset + '/baseline_split/bb_val_mapped.csv', index=False)
        bb_val_e.to_csv('data/' + dataset + '/baseline_split/bb_val_e.csv', index=False)
        print("bb_val saved")
        sh_train.to_csv('data/' + dataset + '/baseline_split/sh_train.csv', index=False)
        sh_train_m.to_csv('data/' + dataset + '/baseline_split/sh_train_mapped.csv', index=False)
        sh_train_e.to_csv('data/' + dataset + '/baseline_split/sh_train_e.csv', index=False)
        print("sh_train saved")
        sh_val.to_csv('data/' + dataset + '/baseline_split/sh_val.csv', index=False)
        sh_val_m.to_csv('data/' + dataset + '/baseline_split/sh_val_mapped.csv', index=False)
        sh_val_e.to_csv('data/' + dataset + '/baseline_split/sh_val_e.csv', index=False)
        print("sh_val saved")
        r2E.to_csv('data/' + dataset + '/baseline_split/r2E.csv', index=False)
        r2E_m.to_csv('data/' + dataset + '/baseline_split/r2E_mapped.csv', index=False)
        r2E_e.to_csv('data/' + dataset + '/baseline_split/r2E_e.csv', index=False)
        print("r2E saved")
        test.to_csv('data/' + dataset + '/baseline_split/test.csv', index=False)
        test_m.to_csv('data/' + dataset + '/baseline_split/test_mapped.csv', index=False)
        test_e.to_csv('data/' + dataset + '/baseline_split/test_e.csv', index=False)
        print("test saved")
    else:
        print("Error in splitted datasets sizes")

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' dataset_name explainer')
    else:
        preprocess(sys.argv[1], sys.argv[2])
