from datetime import datetime
from itertools import product
import mlflow
import multiprocessing as mp
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from util import make_report, model_creation, model_evaluation, model_training, prepareRFdata, prepareNNdata, path, save_obj

sys.path.insert(0, path)

def train(X_train, y_train, X_val, y_val, X_test, y_test, modelType, params, experiment_id, n_classes):
    bb = None
    trained_bb = None
    history = None

    print("Training size: " + str(len(X_train)))
    # Training
    if (modelType == 'NN'):
        # Architecture
        bb = model_creation(params['hidden_layers'], params['hidden_units'], params['act_funct'], params['learning_rate'], params['optimizer'], n_classes)

        # Training and parameters
        trained_bb, history = model_training(bb, X_train, y_train, X_val, y_val, pool_size= None, batch_size=params['batch_size'], epochs = params['epochs'], logdir= None)

    if (modelType == 'RF'):
        # Architecture and parameters
        bb = RandomForestClassifier(bootstrap = params['bootstrap'], max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
        min_samples_leaf = params['min_samples_leaf'], n_estimators = params['n_estimators'], max_features = params['max_features'])

        # Training
        trained_bb = bb.fit(X_train, y_train)

    # Evaluation
    evaluation = model_evaluation(modelType = modelType, model = trained_bb, X_val = X_val, y_val = y_val, X_test = X_test, y_test = y_test)

    # If in top 50 write on MLFlow
    if(float(evaluation['weighted avg-f1-score']) > 0.5 and (float(evaluation['accuracy'] >= 0.75))):
        make_report(modelType = modelType, model = trained_bb, history = history, params = params, metrics = evaluation, experiment_id = experiment_id)

    return trained_bb # relevant metrics

def prepareBBdata(dataset, label, model, final = False):
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
        if(not final):
            bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train_mapped.csv', nrows = 5000)
            bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val_mapped.csv', nrows = 880)
            bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test_mapped.csv', nrows = 1000)
        else:
            bb_train = pd.read_csv('data/' + dataset + '/baseline_split/bb_train_mapped.csv')
            bb_val = pd.read_csv('data/' + dataset + '/baseline_split/bb_val_mapped.csv')
            bb_test = pd.read_csv('data/' + dataset + '/baseline_split/test_mapped.csv')

        prepareData = prepareRFdata

    X_train, y_train = prepareData(bb_train, label)
    X_val, y_val = prepareData(bb_val, label)
    X_test, y_test = prepareData(bb_test, label)

    return X_train, y_train, X_val, y_val, X_test, y_test

def gridSearch(dataset, model, verbose = False):

    grid_params = {}
    label = ''
    n_classes = None

    # MLFlow directory and experiment
    print("Results available at " + mlflow.get_tracking_uri())
    experiment_name = ' '.join([dataset, model])

    if (dataset == 'adult'):
        n_classes = 2
        label = 'class'
        if (model == 'NN'):
            grid_params = {
                'hidden_layers': [1, 2],
                'hidden_units': [2],
                'act_funct': ['relu'],
                'learning_rate': [1e-6],
                'optimizer': [SGD],
                'batch_size': [32],
                'epochs': [200]
            }

        if (model == 'RF'):
            grid_params = {
                'bootstrap': [True, False],
                'max_depth': [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)] + [None],
                'min_samples_split': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'min_samples_leaf': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)],
                'max_features': ['auto', 'sqrt', 0.2, 0.4, 0.6],
                'criterion' : ['gini', 'entropy']
            }

    if (dataset == 'mobility'):
        n_classes = 4
        label = 'class'

        if (model == 'NN'):
            grid_params = {
                'hidden_layers': [1, 2],
                'hidden_units': [12, 24, 40, 50],
                'act_funct': ['relu', 'softmax', 'tanh'],
                'learning_rate': [1e-6, 1e-5, 5e-4, 1e-4],
                'optimizer': [SGD, Adam, RMSprop],
                'batch_size': [None, 16, 32],
                'epochs': [450]
            }

        if (model == 'RF'):
            grid_params = {
                'bootstrap': [True, False],
                'max_depth': [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)] + [None],
                'min_samples_split': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'min_samples_leaf': [int(x) for x in np.linspace(start = 5, stop = 20, num = 4)],
                'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)],
                'max_features': ['auto', 'sqrt', 0.2, 0.4, 0.6],
                'criterion' : ['gini', 'entropy']
            }


    if (dataset == 'texas' or dataset == 'texas_red'):
        n_classes = 100
        label = 'PRINC_SURG_PROC_CODE'
        if (model == 'NN'):
            grid_params = {
                'hidden_layers': [1, 2],
                'hidden_units': [100, 150],
                'act_funct': ['sigmoid', 'tanh'],
                'learning_rate': [1e-6, 1e-5, 1e-7],
                #'optimizer': [Adam, RMSprop],
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
                'max_features': ['auto', 'sqrt', 0.2, 0.4, 0.6],
                'criterion' : ['gini', 'entropy']
            }

    # Listing params combinations
    keys, values = zip(*grid_params.items())
    params_list = [dict(zip(keys, v)) for v in product(*values)]

    # Data
    X_train, y_train, X_val, y_val, X_test, y_test = prepareBBdata(dataset, label, model)

    # Setting MLFlow
    mlflow.set_experiment(experiment_name = experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    # Starting grid search
    print("Running grid search with ",len(params_list)," models")

    # Number of processes
    par_degree = 6

    # Creating one parameters list for each process
    params_lists = [list(a) for a in np.array_split(params_list, par_degree)]

    processes = []

    # Running processes
    for i, params_list in enumerate(params_lists):
        # creare processo
        processes.append(mp.Process(target=train_task, args=(i + 1, params_list, X_train, y_train, X_val, y_val, X_test, y_test, model, exp.experiment_id, n_classes)))

    # Starting grid searches
    for p in processes:
        p.start()

    # Joining
    for p in processes:
        print("Process ", p, " terminated")
        p.join()

    print("Processes joined")

    return True

def train_task(n_process, params_list, X_train, y_train, X_val, y_val, X_test, y_test, model, experiment_id, n_classes):
    for i, params in enumerate(params_list):
        print("Process " + str(n_process) + "- Running train ... (" + str(i + 1) + "/" + str(len(params_list)) + ")")
        train(X_train, y_train, X_val, y_val, X_test, y_test, model, params, experiment_id, n_classes)


if __name__ == "__main__":
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' dataset_name model')
        exit(1)
    dataset = sys.argv[1]
    model_type = sys.argv[2]


    if (dataset not in ['adult', 'mobility', 'texas', 'adult_best', 'texas_red', 'texas_best']):
        print("Unknown dataset")
        exit(1)


    if(dataset.endswith('best')): # Training best model
        experiment_name = ' '.join([dataset, model_type])

        params = None
        n_classes = None
        label = None

        if (dataset == 'texas_best'):
            n_classes = 100
            label = 'PRINC_SURG_PROC_CODE'

            if (model_type == 'RF'):
                params = {
                'bootstrap': False,
                'max_depth': 90,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'n_estimators': 100,
                'max_features': 0.6
                }
            if(model_type == 'NN'):
                pass

        if(dataset == 'adult_best' and model_type == 'RF'):
            n_classes = 2
            label = 'class'
            params = {
            'bootstrap': False,
            'criterion': 'entropy',
            'max_depth': 40,
            'min_samples_split': 5,
            'min_samples_leaf': 5,
            'n_estimators': 100,
            'max_features': 'sqrt'
            }
            pass

        if(dataset == 'adult_best' and model_type == 'NN'):
            pass

        if(dataset == 'mobility_best' and model_type == 'RF'):
            pass

        if(dataset == 'mobility_best' and model_type == 'NN'):
            pass

        # Setting MLFlow
        mlflow.set_experiment(experiment_name = experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)

        # Preparing full data
        print("Preparing data")
        X_train, y_train, X_val, y_val, X_test, y_test = prepareBBdata(dataset.replace('_best', ''), label, model_type, final = True)

        # Training with full data
        print("Training model")
        model = train(X_train, y_train, X_val, y_val, X_test, y_test, model_type, params, exp.experiment_id, n_classes)

        folder = 'data/' + dataset.replace('_best', '') + '/target/' + model_type + '/'
        if(model_type == 'RF'):
            save_obj(model, folder + '/RF_model')

        if(model_type == 'NN'):
            model.save(folder + '/NN_model.h5')

        print("Best model saved in " + folder)

    # else
    else:
        gridSearch(dataset, model_type)
