from abc import ABC, abstractmethod
from attack_util import prepare_target_data
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import load_obj, model_creation, model_training, model_evaluation

class Attack(ABC):

    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        # Target model and data
        self.dataset_name = dataset
        self.n_classes = n_classes
        self.target_labels = {}
        self.target = target
        self.target_train = target_train
        self.target_val = target_val
        self.target_class_name = class_name

        # Shadow properties
        self.n_shadow_models = None
        self.shadow_training = None
        self.shadow_data = None
        self.shadow_train_size = None
        self.shadow_val_size = None
        self.shadow_models = None
        self.shadow_params = None

        # Attack properties
        self.attack_model_type = attack_model_type
        self.attack_models = [None] * self.n_classes
        self.class_indices = {}
        self.X_train_att = None
        self.y_train_att = None
        self.X_val_att = None
        self.y_val_att = None
        self.y_true_attack = None

    @abstractmethod
    def targetPredict(self, X):
        pass

    def getTarget(self):
        return self.target

    def getTargetModel(self):
        return self.target_model

    def getDataset(self):
        return self.dataset_name

    def prepareAttackData(self):

        # Mapping labels
        for i, label in enumerate(sorted(self.shadow_data[self.target_class_name].unique())):
            self.target_labels[i] = label

        # Shaping attack training data
        self.X_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models, self.n_classes))

        self.y_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,1))

        # True labels
        self.y_true_attack = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,))

        # Preparing attack validation
        X_train_target, y_train_target = prepare_target_data(self.target_train, self.target_class_name)
        X_val_target, y_val_target = prepare_target_data(self.target_val, self.target_class_name)

        # Target predictions
        pred_train_target = self.targetPredict(self.target, X_train_target)
        pred_val_target = self.targetPredict(self.target, X_val_target)

        # Balancing predictions on training and validation_data
        idx = np.random.choice(len(pred_train_target), size = len(pred_val_target))
        pred_train_target_sample = pred_train_target[idx]

        # assert len(pred_train_target_sample) == len(pred_val_target)

        # Shaping train and validation predictions
        pred_train_target_shaped = np.array([[np.array(e)] for e in pred_train_target_sample])
        pred_val_target_shaped = np.array([[np.array(e)] for e in pred_val_target])

        # Shaping attack validation data
        self.X_val_att = np.vstack((pred_train_target_shaped, pred_val_target_shaped))

        self.y_val_att = np.zeros(((len(pred_train_target_sample) + len(pred_val_target)), 1))
        self.y_val_att[len(pred_train_target_sample) : len(pred_train_target_sample) + len(pred_val_target)] = 1

        print("Attack data prepared")
        return

    def trainShadowModels(self):

        for i in range(self.n_shadow_models):
            train_shi, val_shi = train_test_split(self.shadow_data,
                                            train_size = self.shadow_train_size,
                                            test_size = self.shadow_val_size,
                                            stratify = shadow_train[self.target_class_name])

            # Training and validation processing
            X_train_shi, y_train_shi = prepare_target_data(train_shi, self.target_class_name)
            X_val_shi, y_val_shi = prepare_target_data(val_shi, self.target_class_name)

            shadow_model = None
            trained_shi = None
            history = None

            # Shadow model creation and training
            if (self.target_model == 'NN'):
                shadow_model = model_creation(shadow_params['hidden_layers'], shadow_params['hidden_units'], shadow_params['act_funct'], shadow_params['learning_rate'], shadow_params['optimizer'], self.n_classes)

                trained_shi, history = model_training(shadow_model, X_train, y_train, X_val, y_val, pool_size= None, batch_size=shadow_params['batch_size'], epochs = shadow_params['epochs'], logdir= None)

            if (self.target_model == 'RF'):
                shadow_model = RandomForestClassifier(bootstrap = shadow_params['bootstrap'], max_depth = shadow_params['max_depth'], min_samples_split = shadow_params['min_samples_split'],
                min_samples_leaf = shadow_params['min_samples_leaf'], n_estimators = shadow_params['n_estimators'], max_features = shadow_params['max_features'])

                trained_shi = shadow_model.fit(X_train_shi, y_train_shi)

            # Model performance
            evaluation = model_evaluation(modelType = self.target_model, model = trained_shi, X_val = X_val_shi, y_val = y_val_shi, X_test = X_val_shi, y_test = y_val_shi)

            print("Shadow model no: %d"%i)
            print('\nFor shadow model with training datasize = ' + str(self.shadow_train_size))
            if (self.target_model == 'NN'):
                print('Training accuracy = %f'%history.history['accuracy'][-1])
            print('Validation accuracy = %f'%evaluation['accuracy'])

            # Saving model
            self.shadow_models[i] = trained_shi

            # Filling attack training data
            ytemp1 = self.targetPredict(trained_shi, X_train_shi)
            ytemp2 = self.targetPredict(trained_shi, X_val_shi)

            self.X_train_att[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.vstack((ytemp1,ytemp2))
            self.y_train_att[i*(self.shadow_train_size + self.shadow_val_size) + self.shadow_train_size : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = 1

            self.y_true_attack[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.hstack((y_train_shi, y_val_shi))

        print("Shadow models trained")
        return

    def trainAttackModel(self):
        # To Do
        for i in range(self.n_classes):
            self.class_indices[i] = [j for j in range(len(self.y_true_attack)) if y_true_attack[j] == self.target_labels[i] ]

        # assert (sum(len(class_indices[i])) == len (y_true_label))

        for i in range(self.n_classes):

            self.attack_models[i] = model_creation(attack_params['hidden_layers'], attack_params['hidden_units'], attack_params['act_funct'], attack_params['learning_rate'], attack_params['optimizer'], self.n_classes)

            trained_shi, history = model_training(self.attack_models[i],
                self.X_train_att[self.class_indices[i]],
                self.y_train_att[self.class_indices[i]],
                self.X_train_att[self.class_indices[i]],
                self.y_train_att[self.class_indices[i]],
                pool_size= None,
                batch_size=attack_params['batch_size'],
                epochs = attack_params['epochs'],
                logdir= None)

        return

    def runAttack(self, shadow_data, shadow_train_size, shadow_val_size, n_shadow_models, shadow_params, attack_params):
        print("Training " + self.dataset_name + " attack")

        self.n_shadow_models = n_shadow_models
        self.shadow_data = shadow_data
        self.shadow_train_size = shadow_train_size
        self.shadow_val_size = shadow_val_size
        self.shadow_params = shadow_params
        self.attack_params = attack_params
        self.shadow_models = [None] * self.n_shadow_models

        print("Preparing attack data")
        self.prepareAttackData()
        print("Attack data prepared")

        print("Training shadow models")
        self.trainShadowModels()
        print("Shadow models trained")

        print("Training attack model(s)")
        self.trainAttackModel()
        print("Attack model(s) trained")

        print("Evaluation...")

        return self.attack_model

class NNAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'NN'

    def targetPredict(self, model, X):
        return model.predict(X)


class RFAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'RF'

    def targetPredict(self, model, X):
        return model.predict_proba(X)


folder = 'data/' + 'texas/'

# Black box
black_box = load_obj(folder + 'target/RF_model')

# Train and val
train_data = pd.read_csv(folder + 'baseline_split/bb_train_mapped.csv', nrows = 10000)
val_data = pd.read_csv(folder + 'baseline_split/bb_val_mapped.csv', nrows = 10000)
shadow_train = pd.read_csv(folder + 'baseline_split/sh_train_mapped.csv', nrows = 10000)

# Shadow paramts
shadow_params = {
    'bootstrap': False,
    'max_depth': 90,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'n_estimators': 100,
    'max_features': 0.6
}

attack_params = {
    'hidden_layers': 1,
    'hidden_units': 100,
    'act_funct': 'sigmoid',
    'learning_rate': 1e-5,
    #'optimizer': [Adam, RMSprop],
    'batch_size': 32,
    'epochs': 200
}

print("Initializing attack")
a = RFAttack('NN', 'texas', black_box, train_data, val_data, 'PRINC_SURG_PROC_CODE', 100)
print("Attack initialized")

print("Attacking...")
a.runAttack(shadow_train, 1000, 1000, 10, shadow_params, attack_params)
