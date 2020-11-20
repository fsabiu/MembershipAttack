from datetime import datetime
from itertools import product
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
from tensorflow.keras.optimizers import Adam, SGD
from util import make_report, model_creation, model_evaluation, model_training, prepareRFdata, prepareNNdata, path

sys.path.insert(0, path)

def train(X_train, y_train, X_val, y_val, X_test, y_test, model, params, logdir):
    bb = None
    trained_bb = None

    # Training
    if (model == 'NN'):
        # Architecture
        bb = model_creation(params['hidden_layers'], params['hidden_units'], params['act_funct'], params['learning_rate'], params['optimizer'])

        # Training and parameters
        trained_bb = model_training(bb, X_train, y_train, X_val, y_val, pool_size=params['pool_size'], batch_size=params['batch_size'], epochs = params['epochs'], logdir=logdir)

    if (model == 'RF'):
        # Architecture and parameters
        bb = RandomForestClassifier(bootstrap = params['bootstrap'], max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
        min_samples_leaf = params['min_samples_leaf'], n_estimators = params['n_estimators'], max_features = params['max_features'])

        # Training
        trained_bb = bb.fit(X_train, y_train)

    # Evaluation
    evaluation = model_evaluation(modelType = model, model = trained_bb, X_test = X_test, y_test = y_test)

    # Write on TensorBoard
    make_report(model, params, evaluation, logdir)

    return True # relevant metrics

def prepareBBdata(dataset, label, model):
    """
        Returns the dataset for the training

        Parameters:
        dataset (String): Dataset name
        dataset (String): Dataset label name
        model (String): model name

        Returns:
        X_train, y_train, X_val, y_val, X_test, y_test

    """
    bb_train = bb_val = bb_test = None
    prepareData = None

    if (model == 'NN'):
        # Data
        bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train_e.csv')
        bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val_e.csv')
        bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test_e.csv')

        prepareData = prepareNNdata

    if (model == 'RF'):
        # Data
        bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train_e.csv')
        bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val_e.csv')
        bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test_e.csv')

        prepareData = prepareRFdata

    if(dataset == 'texas'):
    
        if (model == 'NN'):
            # Data
            bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train.csv')
            bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val.csv')
            bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test.csv')

        if (model == 'RF'):
            # Data
            bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train_mapped.csv', nrows = 5000)
            bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val_mapped.csv', nrows = 880)
            bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test_mapped.csv', nrows = 1000)

    X_train, y_train = prepareData(bb_train, label)
    X_val, y_val = prepareData(bb_val, label)
    X_test, y_test = prepareData(bb_test, label)

    return X_train, y_train, X_val, y_val, X_test, y_test

def gridSearch(dataset, model, verbose = False):
    
    grid_params = {}
    label = ''
    logdir = path + '/data/' + dataset + '/target/' + model + '/'

    if (dataset == 'adult'):
        label = 'class'
        if (model == 'NN'):
            grid_params = {
                'hidden_layers': [1],
                'hidden_units': [2],
                'act_funct': ['relu'],
                'learning_rate': [1e-6],
                'optimizer': [SGD],
                'batch_size': [32],
                'epochs': [200]
            }
        
        if (model == 'RF'):
            """grid_params = {
                'bootstrap': [True, False],
                'max_depth': [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)] + [None],
                'min_samples_split': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'min_samples_leaf': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)],
                'max_features': ['auot', 'sqrt', 0.2, 0.4, 0.6]
            }"""
            grid_params = {
                'bootstrap': [True],
                'max_depth': [10],
                'min_samples_split': [4],
                'min_samples_leaf': [4],
                'n_estimators': [5],
                'max_features': ['sqrt']
            }

    if (dataset == 'texas'):
        label = 'PRINC_SURG_PROC_CODE'
        if (model == 'NN'):
            grid_params = {
                'hidden_layers': [1, 2],
                'hidden_units': [100, 150],
                'act_funct': ['relu', 'tanh'],
                'learning_rate': [1e-6, 1e-5, 1e-7],
                #'optimizer': [Adam, RMSprop],
                'pool_size': [1, 8],
                'batch_size': [None, 32],
                'epochs': [200, 300]
            }
        
        if (model == 'RF'):
            grid_params = {
                'bootstrap': [True, False],
                'max_depth': [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)] + [None],
                'min_samples_split': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'min_samples_leaf': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)],
                'max_features': ['auot', 'sqrt', 0.2, 0.4, 0.6],
                'criterion' : ['gini', 'entropy']
            }

    # Listing params combinations
    keys, values = zip(*grid_params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test = prepareBBdata(dataset, label, model)

    print("Running grid search with ",len(params_list)," models")
    for i, params in enumerate(params_list):
        date = datetime.now().strftime("%Y%m%d_%H%M%S")+"/"
        print("Running train ... (" + str(i + 1) + "/" + str(len(params_list) + ")"))
        res = train(X_train, y_train, X_val, y_val, X_test, y_test, model, params, logdir + str(date))

    return True

if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' dataset_name model')
        exit(1)
    if (sys.argv[1] not in ['adult', 'mobility', 'texas']):
        print("Unknown dataset")
        exit(1)
    
    # else
    gridSearch(sys.argv[1], sys.argv[2])