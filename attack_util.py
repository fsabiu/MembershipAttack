import numpy as np
import pandas as pd
import tensorflow as tf

from attack import Attack, NNAttack, RFAttack
from util import load_obj, save_obj

def call_attack(dataset, model_type, shadow_params, attack_params, class_name, n_classes, shadow_train_size, shadow_val_size, n_shadow_models):
    folder = 'data/' + dataset + '/'
    black_box = None

    # Black box
    if( model_type == 'RF'):
        black_box = load_obj(folder + 'target/' + model_type +'/RF_model')
    if( model_type == 'NN'):
        black_box = tf.keras.models.load_model(folder + 'target/' + model_type +'/NN_model.h5')

    # Train and val
    train_data = None
    val_data = None
    shadow_train = None

    if(model_type == 'RF'):
        if(dataset == 'texas'):
            train_data = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv', nrows = shadow_train_size)
            val_data = pd.read_csv(folder + 'baseline_split/bb_val_mapped.csv', nrows = shadow_train_size)
            shadow_train = pd.read_csv(folder + 'baseline_split/sh_train_mapped.csv', nrows = 100000)
        else:
            train_data = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv')
            val_data = pd.read_csv(folder + 'baseline_split/bb_val_mapped.csv')
            shadow_train = pd.read_csv(folder + 'baseline_split/sh_train_mapped.csv')


    if(model_type == 'NN'):
        train_data = pd.read_csv(folder + 'baseline_split/bb_train_e.csv')
        val_data = pd.read_csv(folder + 'baseline_split/bb_val_e.csv')
        shadow_train = pd.read_csv(folder + 'baseline_split/sh_train_e.csv')

    # Initializing attack
    attack = None

    if(model_type == 'RF'):
        attack = RFAttack(attack_model_type = 'RF',
            dataset = dataset,
            target = black_box,
            target_train = train_data,
            target_val = val_data,
            class_name = class_name,
            n_classes = n_classes)

    if(model_type == 'NN'):
        attack = NNAttack(attack_model_type = 'NN',
            dataset = dataset,
            target = black_box,
            target_train = train_data,
            target_val = val_data,
            class_name = class_name,
            n_classes = n_classes)

    models, histories = attack.runAttack(
        shadow_train,
        shadow_train_size,
        shadow_val_size,
        n_shadow_models,
        shadow_params,
        attack_params)

    return models, histories

def prepare_target_data(dataset, class_name):
    y = dataset[class_name].values
    dataset.drop([class_name], axis=1, inplace = True)

    cols = dataset.columns
    x = np.array(dataset[cols])

    return x, y

def writeAttackModels(dataset, models, histories):
    folder = 'data/' + dataset + '/results/'

    for i, model in enumerate(models):
        model.save(folder + 'attack_model_' + str(i) + '.h5')
        #save_obj(histories[i], 'attack_history_' + str(i))
