from attack import *
from util import path

import sys
sys.path.insert(0, path)

def call_attack(dataset, model_type):
    folder = 'data/' + dataset + '/'

    # Black box
    black_box = load_obj(folder + 'target/RF/RF_model')

    # Train and val
    train_data = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv', nrows = 10000)
    val_data = pd.read_csv(folder + 'baseline_split/bb_val_mapped.csv', nrows = 10000)
    shadow_train = pd.read_csv(folder + 'baseline_split/sh_train_mapped.csv', nrows = 50000)

    # Shadow params
    shadow_params = {
        'bootstrap': False,
        'criterion': 'entropy',
        'max_depth': 40,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'n_estimators': 100,
        'max_features': 'sqrt'
    }

    attack_params = {
        'hidden_layers': 1,
        'hidden_units': 150,
        'act_funct': 'sigmoid',
        'learning_rate': 1e-2,
        'optimizer': Adam,
        'batch_size': 32,
        'epochs': 200
    }

    # Initializing attack
    attack = RFAttack(attack_model_type = 'NN',
        dataset = dataset,
        target = black_box,
        target_train = train_data,
        target_val = val_data,
        class_name = 'class',
        n_classes = 2)

    attack_model = attack.runAttack(
        shadow_train,
        10000,
        10000,
        10,
        shadow_params,
        attack_params)

if __name__ == "__main__":

    if(len(sys.argv) != 2):
        print('Usage: ' + sys.argv[0] + ' model')
        exit(1)

    if (model not in ['RF', 'NN']):
        print("Model not implemented")
        exit(1)

    model = sys.argv[1]
    dataset = 'adult'

    call_attack(dataset, model)
