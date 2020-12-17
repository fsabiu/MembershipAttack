from explanator import *
from lime.lime import lime_tabular
from util import encode_dataset, load_obj
import pandas as pd
import sklearn
import sys

def adult_explain(model_type, bb, r2E):

    n_classes = 2

    train = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv')

    dataframe = pd.concat([train, r2E], axis=0)

    print(len(dataframe) == len(train) + len(r2E))
    # LORE function 'encoded_dataset' -> always returns numeric columns
    #encoded_data, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = encode_dataset(dataframe, 'class')

    categorical_features = [1, 2, 3, 4, 5, 6, 7, 11, 12]
    feature_names = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    class_names = dataframe['class'].unique()

    data = dataframe.to_numpy()
    train_data = train.to_numpy()
    #r2E_data = r2E.drop(['class'], axis = 1)
    r2E_data = r2E.to_numpy()

    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_

    data = data.astype(float)
    train_data = train_data.astype(float)
    r2E_data = r2E_data.astype(float)

    ex = lime_tabular.LimeTabularExplainer(
            data,
            class_names = class_names,
            feature_names = feature_names,
            categorical_features = categorical_features,
            categorical_names = categorical_names,
            kernel_width = 3,
            verbose = False,
            discretize_continuous = False)


    predict_fn = lambda x: bb.predict_proba(x[:12]).astype(float)

    neighs_dfs = []
    for i in range(len(r2E)):
        if(i%1000 == 0):
            print('Progress: ' + str(i) + '/' + str(len(r2E)))

        data_exp, inverse = ex.explain_instance(r2E_data[i], predict_fn, num_features=5)
        neighs_dfs.append(pd.DataFrame(data = inverse, columns = r2E.columns).astype(r2E.dtypes))

    neighs = pd.concat(neighs_dfs, axis = 0)

    if (len(neighs) == 5000*len(r2E)):
        print("Size checking done")

    return neighs

def mobility_explain(model_type, bb, r2E):

    n_classes = 4

    train = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv')

    dataframe = pd.concat([train, r2E], axis=0)

    print(len(dataframe) == len(train) + len(r2E))
    # LORE function 'encoded_dataset' -> always returns numeric columns
    #encoded_data, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = encode_dataset(dataframe, 'class')

    print(train.columns)
    print(train.head(5))

    categorical_features = [24]
    feature_names = ['max_distance_from_home', 'maximum_distance', 'max_tot',
       'distance_straight_line', 'sld_avg', 'radius_of_gyration',
       'norm_uncorrelated_entropy', 'number_of_visits', 'nv_avg',
       'number_of_locations', 'nlr', 'raw_home_freq', 'raw_work_freq',
       'raw_least_freq', 'home_freq_avg', 'work_freq_avg', 'hf_tot_df',
       'wf_tot_df', 'n_user_home', 'n_user_work', 'n_user_home_avg',
       'n_user_work_avg', 'home_entropy', 'work_entropy', 'class']
    class_names = dataframe['class'].unique()

    data = dataframe.to_numpy()
    train_data = train.to_numpy()
    #r2E_data = r2E.drop(['class'], axis = 1)
    r2E_data = r2E.to_numpy()

    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_

    data = data.astype(float)
    train_data = train_data.astype(float)
    r2E_data = r2E_data.astype(float)

    ex = lime_tabular.LimeTabularExplainer(
            data,
            class_names = class_names,
            feature_names = feature_names,
            categorical_features = categorical_features,
            categorical_names = categorical_names,
            kernel_width = 3,
            verbose = False,
            discretize_continuous = False)


    predict_fn = lambda x: bb.predict_proba(x).astype(float)

    neighs_dfs = []
    for i in range(len(r2E)):
        if(i%1000 == 0):
            print('Progress: ' + str(i) + '/' + str(len(r2E)))

        data_exp, inverse = ex.explain_instance(r2E_data[i], predict_fn, num_features=5)
        neighs_dfs.append(pd.DataFrame(data = inverse, columns = r2E.columns).astype(r2E.dtypes))

    neighs = pd.concat(neighs_dfs, axis = 0)

    if (len(neighs) == 5000*len(r2E)):
        print("Size checking done")

    return neighs


if __name__ == "__main__":
    """
    Requires:
    - data/dataset/results/attack_models/
    to exists
    """
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' model dataset')
        exit(1)

    # Target params
    model = sys.argv[1]
    dataset = sys.argv[2]

    if (model not in ['RF', 'NN']):
        print("Model not implemented")
        exit(1)

    if (dataset not in ['adult', 'mobility', 'texas']):
        print("Dataset not supported")
        exit(1)

    folder = 'data/' + dataset + '/'

    r2E = pd.read_csv(folder + 'baseline_split/r2E_mapped.csv')
    bb = None

    neighs = None

    if (model == 'RF'):
        bb = load_obj(folder + 'target/' + model + '/RF_model')

    if(model == 'NN'):
        bb = load_obj(folder + 'target/' + model + '/RF_model')

    if(dataset == 'adult' and model == 'RF'):
        neighs = adult_explain(model, bb, r2E)

    if(dataset == 'mobility'):
        neighs = mobility_explain(model, bb, r2E)

    if(dataset == 'texas'):
        neighs = texas_explain(model, bb, r2E)

    neighs.to_csv(folder + 'baseline_split/r2E_mapped_explained.csv')
    print("Neighbors written")
