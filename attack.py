from abc import ABC, abstractmethod
from attack_util import prepare_target_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import model_creation, model_training, model_evaluation

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
        self.attack_models = None
        self.X_train_att = None
        self.y_train_att = None
        self.X_val_att = None
        self.y_val_att = None
        self.X_true_att = None

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
        for i, label in enumerate(self.dataset[self.class_name].unique()):
            self.target_labels[i] = label

        # Shaping attack training data
        self.X_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,1))

        self.y_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,1))
        self.X_true_att = np.zeros(((self.shadow_train_size + self.shadow_val_size) * self.n_shadow_models,))

        # Preparing attack validation
        X_train_target, y_train_target = prepare_target_data(self.target_train)
        X_val_target, y_val_target = prepare_target_data(self.target_val)

        # Target predictions
        pred_train_target = self.targetPredict(X_train_target)
        pred_val_target = self.targetPredict(X_val_target)

        # Balancing predictions on training and validation_data
        pred_train_target_sample = np.random.choice(pred_train_target, len(pred_val_target))

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
            X_train_shi, y_train_shi = prepare_target_data(self.target_class_name)
            X_val_shi, y_val_shi = prepare_target_data(self.target_class_name)

            shadow_model = None
            trained_shi = None
            history = None

            # Shadow model creation and training
            if (target_model == 'NN'):
                shadow_model = model_creation(shadow_params['hidden_layers'], shadow_params['hidden_units'], shadow_params['act_funct'], shadow_params['learning_rate'], shadow_params['optimizer'], self.n_classes)

                trained_shi, history = model_training(shadow_model, X_train, y_train, X_val, y_val, pool_size= None, batch_size=shadow_params['batch_size'], epochs = shadow_params['epochs'], logdir= None)

            if (target_model == 'RF'):
                shadow_model = RandomForestClassifier(bootstrap = shadow_params['bootstrap'], max_depth = shadow_params['max_depth'], min_samples_split = shadow_params['min_samples_split'],
                min_samples_leaf = shadow_params['min_samples_leaf'], n_estimators = shadow_params['n_estimators'], max_features = shadow_params['max_features'])

                trained_shi = shadow_model.fit(X_train, y_train)

            # Model performance
            evaluation = model_evaluation(modelType = target_model, model = trained_shi, X_val = X_val_shi, y_val = y_val_shi, X_test = X_val_shi, y_test = y_val_shi)

            print("Shadow model no: %d"%i)
            print('\nFor shadow model with training datasize = ' + str(self.shadow_train_size))
            if (target_model == 'NN'):
                print('Training accuracy = %f'%history.history['accuracy'][-1])
            print('Validation accuracy = %f'%evaluation['accuracy'][-1])

            # Saving model
            self.shadow_models[i] = trained_shi

            # Filling attack training data
            ytemp1 = trained_shi.predict(X_train_shi)
            ytemp2 = trained_shi.predict(X_val_shi)

            self.X_train_att[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.vstack((ytemp1,ytemp2))
            self.y_train_att[i*(self.shadow_train_size + self.shadow_val_size) + self.shadow_train_size : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = 1

            self.X_true_att[i*(self.shadow_train_size + self.shadow_val_size) : (i+1) * (self.shadow_train_size + self.shadow_val_size)] = np.hstack((y_train_shi, y_val_shi))

        print("Shadow models trained")
        return

    @abstractmethod
    def trainAttackModel(self):
        # To Do
        for i in self.n_classes:
            # Model creation and training
            if (target_model == 'NN'):
                shadow_model = model_creation(shadow_params['hidden_layers'], shadow_params['hidden_units'], shadow_params['act_funct'], shadow_params['learning_rate'], shadow_params['optimizer'], self.n_classes)

                trained_shi, history = model_training(shadow_model, X_train, y_train, X_val, y_val, pool_size= None, batch_size=shadow_params['batch_size'], epochs = shadow_params['epochs'], logdir= None)

            if (target_model == 'RF'):
                shadow_model = RandomForestClassifier(bootstrap = shadow_params['bootstrap'], max_depth = shadow_params['max_depth'], min_samples_split = shadow_params['min_samples_split'],
                min_samples_leaf = shadow_params['min_samples_leaf'], n_estimators = shadow_params['n_estimators'], max_features = shadow_params['max_features'])

                trained_shi = shadow_model.fit(X_train, y_train)

        return

    def runAttack(self, shadow_data, shadow_train_size, shadow_val_size, n_shadow_models, shadow_params, attack_params):
        print("Training " + self.dataset_name + " attack")

        self.n_shadow_models = n_shadow_models
        self.shadow_data = shadow_data
        self.shadow_train_size = shadow_train_size
        self.shadow_val_size = shadow_val_size
        self.shadow_params = shadow_params
        self.attack_params = attack_params
        self.shadow_models = [None * self.n_shadow_models]

        self.prepareAttackData()

        self.trainShadowModels()

        self.trainAttackModel()

        return self.attack_model

class NNAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'NN'

    def targetPredict(self, X):
        return self.target.predict(X)

    def trainAttackModel(self):
        for i in self.n_classes:
            # Model creation and training
                shadow_model = model_creation(shadow_params['hidden_layers'], shadow_params['hidden_units'], shadow_params['act_funct'], shadow_params['learning_rate'], shadow_params['optimizer'], self.n_classes)

                trained_shi, history = model_training(shadow_model, X_train, y_train, X_val, y_val, pool_size= None, batch_size=shadow_params['batch_size'], epochs = shadow_params['epochs'], logdir= None)

                trained_shi = shadow_model.fit(X_train, y_train)

class RFAttack(Attack):
    def __init__(self, attack_model_type, dataset, target, target_train, target_val, class_name, n_classes):
        super().__init__(attack_model_type, dataset, target, target_train, target_val, class_name, n_classes)
        self.target_model = 'RF'

    def targetPredict(self, X):
        return self.target.predict(X)

    def trainAttackModel(self):
        for i in self.n_classes:
            # Model creation and training
                shadow_model = model_creation(shadow_params['hidden_layers'], shadow_params['hidden_units'], shadow_params['act_funct'], shadow_params['learning_rate'], shadow_params['optimizer'], self.n_classes)

                shadow_model = RandomForestClassifier(bootstrap = shadow_params['bootstrap'], max_depth = shadow_params['max_depth'], min_samples_split = shadow_params['min_samples_split'],
                min_samples_leaf = shadow_params['min_samples_leaf'], n_estimators = shadow_params['n_estimators'], max_features = shadow_params['max_features'])

                trained_shi = shadow_model.fit(X_train, y_train)

a = Attack('NN', 'adult', 'bb', 'RF', 'x_train', 'x_val', 'class', 2)
a.runAttack('shadow_data', 100, 10, 'target_params', 'attack_params')

m = Attack('NN', 'mobility', 'bb', 'RF', 'x_train', 'x_val', 'class', 4)
