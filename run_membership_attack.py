from util import load_obj, path
from attack_util import call_attack, writeAttackModels
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

import sys
sys.path.insert(0, path)

def adult_attack(model):
    dataset = 'adult'
    class_name = 'class'
    n_classes = 2

    # Attack params
    shadow_train_size = 5000
    shadow_val_size = 5000
    n_shadow_models = 2

    # Shadow params
    shadow_params = None

    if(model == 'RF'):
        shadow_params = {
            'bootstrap': False,
            'criterion': 'entropy',
            'max_depth': 40,
            'min_samples_split': 5,
            'min_samples_leaf': 5,
            'n_estimators': 100,
            'max_features': 'sqrt'
        }

    if(model == 'NN'):
        shadow_params = {
            'hidden_layers': 2,
            'hidden_units': 106,
            'act_funct': 'tanh',
            'learning_rate': 1e-5,
            'optimizer': RMSprop,
            'batch_size': 16,
            'epochs': 200,
            'loss': 'BinaryCrossentropy'
        }

    attack_params = {
        'hidden_layers': 1,
        'hidden_units': 150,
        'act_funct': 'sigmoid',
        'learning_rate': 1e-2,
        'optimizer': Adam,
        'batch_size': 32,
        'epochs': 200,
        'loss': 'BinaryCrossentropy'
    }

    models, histores = call_attack(
                    dataset = dataset,
                    model_type = model,
                    shadow_params = shadow_params,
                    attack_params = attack_params,
                    class_name = class_name,
                    n_classes = n_classes,
                    shadow_train_size = shadow_train_size,
                    shadow_val_size = shadow_val_size,
                    n_shadow_models = n_shadow_models)

    print("Done")
    return models, histores

def mobility_attack(model):
    dataset = 'mobility'
    class_name = 'class'
    n_classes = 4

    # Attack params
    shadow_train_size = 1200
    shadow_val_size = 1200
    n_shadow_models = 4

    shadow_params = None

    if(model == 'NN'):
        shadow_params = {
            'hidden_layers': 2,
            'hidden_units': 50,
            'act_funct': 'tanh',
            'learning_rate': 1e-4,
            'optimizer': RMSprop,
            'batch_size': 32,
            'epochs': 450,
            'loss': 'CategoricalCrossentropy'
        }

    if(model == 'RF'):
        shadow_params = {
            'bootstrap': True,
            'criterion': 'entropy',
            'max_depth': 160,
            'min_samples_split': 20,
            'min_samples_leaf': 5,
            'n_estimators': 100,
            'max_features': 'auto'
        }

    attack_params = {
        'hidden_layers': 1,
        'hidden_units': 150,
        'act_funct': 'sigmoid',
        'learning_rate': 1e-2,
        'optimizer': Adam,
        'batch_size': 8,
        'epochs': 200,
        'loss': 'BinaryCrossentropy'
    }

    models, histores = call_attack(
                    dataset = dataset,
                    model_type = model,
                    shadow_params = shadow_params,
                    attack_params = attack_params,
                    class_name = class_name,
                    n_classes = n_classes,
                    shadow_train_size = shadow_train_size,
                    shadow_val_size = shadow_val_size,
                    n_shadow_models = n_shadow_models)

    return models, histores

def texas_attack(model):
    dataset = 'texas'
    class_name = 'PRINC_SURG_PROC_CODE'
    n_classes = 100

    # Attack params
    shadow_train_size = 15000
    shadow_val_size = 15000
    n_shadow_models = 1

    # Shadow params
    shadow_params = {
        'bootstrap': False,
        'max_depth': 150,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'n_estimators': 522,
        'max_features': 0.4
    }

    attack_params = {
        'hidden_layers': 1,
        'hidden_units': 150,
        'act_funct': 'sigmoid',
        'learning_rate': 1e-3,
        'optimizer': Adam,
        'batch_size': 32,
        'epochs': 200,
        'loss': 'BinaryCrossentropy'
    }

    models, histores = call_attack(
                    dataset = dataset,
                    model_type = model,
                    shadow_params = shadow_params,
                    attack_params = attack_params,
                    class_name = class_name,
                    n_classes = n_classes,
                    shadow_train_size = shadow_train_size,
                    shadow_val_size = shadow_val_size,
                    n_shadow_models = n_shadow_models)

    return models, histores

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

    # Results
    models = histories = None

    if(dataset == 'adult'):
        models, histories = adult_attack(model)

    if(dataset == 'mobility'):
        models, histories = mobility_attack(model)

    if(dataset == 'texas'):
        models, histories = texas_attack(model)

    writeAttackModels(dataset, 'NN', models, histories)
