import dask.dataframe as dd
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import sys
from util import encode_dataset, encode_Dask_dataset, encode_split_dataset, get_unique_df_values, load_obj, map_columns, path, prepare_dataset, save_obj, split

sys.path.insert(0, path)

def preprocess(dataset, explainer):

    # Dataset preparation
    data =  class_name = None
    dtypes = {}
    encoded_data = None

    if (dataset == 'texas'): #
        class_name = 'PRINC_SURG_PROC_CODE'
        dtypes = load_obj('data/' + dataset + '/dtypes')
        data = pd.read_csv('data/' + dataset + '/' + dataset +'_mapped.csv', dtype = dtypes)

        columns2remove = ['RECORD_ID', 'PRINC_ICD9_CODE']
        data.drop(columns2remove, inplace=True, axis=1)

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

    else:
        data, class_name = prepare_dataset(dataset, explainer)
        # Mapping
        mapped_data = map_columns(data, class_name)
        mapped_data.to_csv('data/' + dataset + '/' + dataset + '_mapped.csv')

    # Encoding
    if(dataset == 'adult'):
        class_name = 'class'
        for col in data.columns:
            if(col in ['capital-gain', 'capital-loss']):
                dtypes[col] = 'float32'
            elif(col in ['age', 'hours-per-week']):
                dtypes[col] = 'int64'
            else:
                dtypes[col] = 'object'

    if(dataset == 'mobility'):
        class_name = 'class'
        for col in data.columns:
            if(col in ['max_distance_from_home', 'maximum_distance', 'max_tot', 'distance_straight_line',
            'sld_avg', 'radius_of_gyration', 'norm_uncorrelated_entropy', 'nlr', 'home_freq_avg',
            'work_freq_avg', 'hf_tot_df', 'wf_tot_df', 'n_user_home_avg', 'n_user_work_avg', 'home_entropy',
            'work_entropy']):
                dtypes[col] = 'float32'
            elif( col in ['uid', 'wait', 'number_of_visits', 'nv_avg', 'number_of_locations',
            'raw_home_freq', 'raw_work_freq', 'raw_least_freq', 'n_user_home', 'n_user_work']):
                dtypes[col] = 'int64'
            else:
                dtypes[col] = 'object'

    encoded_data = encode_Dask_dataset(dd.from_pandas(data, npartitions = 1), class_name, dtypes, [])
    #encoded_data, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = encode_dataset(data, class_name)
    encoded_data.to_csv('data/' + dataset + '/' + dataset + '_encoded.csv')

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
