from abc import ABC, abstractmethod
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from util import load_obj, make_report, model_creation, model_training, model_evaluation, prepareNNdata, prepareRFdata

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
        self.attack_histories = [None] * self.n_classes
        self.class_indices_train = []
        self.class_indices_val = []
        self.X_train_att = None
        self.y_train_att = None
        self.X_val_att = None
        self.y_val_att = None
        self.y_true_attack = None

    @abstractmethod
    def targetPredict(self, X):
        pass

    def getAttackModel(self):
        return self.attack_models

    def getAttackTrain(self):
        return self.X_train_att, self.y_train_att

    def getClassIndices(self):
        return self.class_indices_train, self.class_indices_val

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

        #save_obj(self.target_labels, '')

        # Shaping attack training data
        self.X_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models, self.n_classes))
        self.y_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,1))

        # True labels
        self.y_true_attack = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,))

        # Preparing attack validation
        X_train_target, y_train_target = self.prepare_target_data(self.target_train, self.target_class_name)
        X_val_target, y_val_target = self.prepare_target_data(self.target_val, self.target_class_name)

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

        # Getting classes from predictions
        y_train_target_class = np.array([self.target_labels[np.argmax(vect)] for vect in y_train_target])
        y_val_target_class = np.array([self.target_labels[np.argmax(vect)] for vect in y_val_target])

        self.y_val_true =  np.hstack((y_train_target_class[idx], y_train_target_class))

        return

    def trainShadowModels(self):

        for i in range(self.n_shadow_models):
            train_shi, val_shi = train_test_split(self.shadow_data,
                                            train_size = self.shadow_train_size,
                                            test_size = self.shadow_val_size,
                                            stratify = self.shadow_data[self.target_class_name])

            # Training and validation processing
            X_train_shi, y_train_shi = self.prepare_target_data(train_shi, self.target_class_name)
            X_val_shi, y_val_shi = self.prepare_target_data(val_shi, self.target_class_name)

            shadow_model = None
            trained_shi = None
            history = None

            # Shadow model creation and training
            if (self.target_model == 'NN'):
                shadow_model = model_creation(
                self.shadow_params['hidden_layers'],
                self.shadow_params['hidden_units'],
                self.shadow_params['act_funct'],
                self.shadow_params['learning_rate'],
                self.shadow_params['optimizer'],
                self.shadow_params['loss'],
                self.n_classes)

                trained_shi, history = model_training(shadow_model, X_train_shi, y_train_shi, X_val_shi, y_val_shi, pool_size= None, batch_size=self.shadow_params['batch_size'], epochs = self.shadow_params['epochs'], logdir= None)

            if (self.target_model == 'RF'):
                shadow_model = RandomForestClassifier(bootstrap = self.shadow_params['bootstrap'], max_depth = self.shadow_params['max_depth'], min_samples_split = self.shadow_params['min_samples_split'],
                min_samples_leaf = self.shadow_params['min_samples_leaf'], n_estimators = self.shadow_params['n_estimators'], max_features = self.shadow_params['max_features'])

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

            ytemp1_class = np.array([np.argmax(vect) for vect in ytemp1])
            ytemp2_class = np.array([np.argmax(vect) for vect in ytemp2])

            self.X_train_att[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.vstack((ytemp1,ytemp2))
            self.y_train_att[i*(self.shadow_train_size + self.shadow_val_size) + self.shadow_train_size : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = 1

            self.y_true_attack[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.hstack((ytemp1_class, ytemp2_class))

        print("Shadow models trained")
        return

    def trainAttackModel(self):

        for i in range(self.n_classes):
            # self.class_indices_train[i] contains indices of X_train_att corresponding to class self.target_labels[i]
            self.class_indices_train.append([j for j in range(len(self.X_train_att)) if self.y_true_attack[j] == self.target_labels[i] ])

            # self.class_indices_val[i] contains indices of X_val_att corresponding to class self.target_labels[i]
            self.class_indices_val.append([j for j in range(len(self.X_val_att)) if self.y_val_true[j] == self.target_labels[i] ])

        # Assert sizes of mapping data -> original class
        #print(sum([len(idces) for idces in self.class_indices_train]) == len(self.y_true_attack))
        #print(sum([len(idces) for idces in self.class_indices_val]) == len(self.y_val_true))

        # Setting MLFlow
        experiment_name = "attack " + self.dataset_name + " " + self.target_model
        mlflow.set_experiment(experiment_name = experiment_name)
        exp = mlflow.get_experiment_by_name(experiment_name)

        for i in range(self.n_classes):
            self.attack_models[i] = model_creation(
            hidden_layers = self.attack_params['hidden_layers'],
            hidden_units = self.attack_params['hidden_units'],
            act_function = self.attack_params['act_funct'],
            learning_rate = self.attack_params['learning_rate'],
            optimizer = self.attack_params['optimizer'],
            loss = self.attack_params['loss'],
            output_units = 1,
            input_size = self.n_classes)

            # Permuting train and test
            p_train = np.random.permutation(len(self.class_indices_train[i]))
            p_val = np.random.permutation(len(self.class_indices_val[i]))

            # Defining train and test
            X_train = self.X_train_att[self.class_indices_train[i]][p_train]
            y_train = self.y_train_att[self.class_indices_train[i]][p_train]
            X_val = self.X_val_att[self.class_indices_val[i]][p_val]
            y_val = self.y_val_att[self.class_indices_val[i]][p_val]

            print("Training attack for class " + str(self.target_labels[i]) + "with train: " + str(len(X_train)) + ", val = " + str(len(X_val)))
            self.attack_models[i], self.attack_histories[i] = model_training(self.attack_models[i],
                X_train,
                y_train,
                X_val,
                y_val,
                pool_size = None,
                batch_size = self.attack_params['batch_size'],
                epochs = self.attack_params['epochs'],
                logdir = None)

            # Evaluation
            print("Train shapesss")
            print(np.shape(X_train))
            print(np.shape(X_val))

            evaluation = model_evaluation(modelType = 'NN_attack', model = self.attack_models[i], X_val = X_val, y_val = y_val, X_test = X_val, y_test = y_val)

            # Logs
            self.attack_params['target_class'] = self.target_labels[i]
            make_report(modelType = 'NN', model = self.attack_models[i], history = self.attack_histories[i], params = self.attack_params, metrics = evaluation, experiment_id = exp.experiment_id)

        return self.attack_models, self.attack_histories

    def runAttack(self, shadow_data, shadow_train_size, shadow_val_size, n_shadow_models, shadow_params, attack_params):
        # Checking sizes
        if(len(shadow_data) < shadow_train_size + shadow_val_size):
            raise Exception("Error: Shadow models dataset sizes. Total available records: " + str(len(shadow_data)))

        if(n_shadow_models != self.n_classes):
            print("Warning ... Parameters not optimized")

        print("Starting " + self.dataset_name + " attack")

        self.n_shadow_models = n_shadow_models
        self.shadow_data = shadow_data
        self.shadow_train_size = shadow_train_size
        self.shadow_val_size = shadow_val_size
        self.shadow_params = shadow_params
        self.attack_params = attack_params
        self.shadow_models = [None] * self.n_shadow_models

        print("Preparing attack data")
        y_values = self.prepareAttackData()
        print("Attack data prepared")

        print("Training shadow models")
        self.trainShadowModels()
        print("Shadow models trained")

        print("Training attack model(s)")
        models, histories = self.trainAttackModel()
        print("Attack model(s) trained")

        return models, histories

class NNAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'NN'
        self.prepare_target_data = prepareNNdata

    def targetPredict(self, model, X):
        return model.predict(X)


class RFAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'RF'
        self.prepare_target_data = prepareRFdata

    def targetPredict(self, model, X):
        return model.predict_proba(X)
