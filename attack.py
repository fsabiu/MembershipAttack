from attack_util import prepare_target_data
import numpy as np

class Attack(object):

    def __init__(self, attack_model, dataset, target, target_model, target_train, target_val, class_name):
        # Initializations
        self.dataset_name = dataset
        self.target = target
        self.target_model = target_model
        self.target_train = target_train
        self.target_val = target_val
        self.target_class_name = class_name

        # Shadow properties
        self.shadow_training = None
        self.shadow_data = None
        self.shadow_train_size = None
        self.shadow_val_size = None
        self.shadow_params = None
        self.n_shadow_models = None

        # Attack properties
        self.attack_model = attack_model
        self.X_train_att = None
        self.y_train_att = None
        self.X_val_att = None
        self.y_val_att = None
        self.y_true_att = None

    def targetPredict(self, X):
        if(self.target_model == 'NN'):
            return self.target.predict(X)

        if(self.target_model == 'RF'):
            return self.target.predict(X)

    def getTarget(self):
        return self.target

    def getTargetModel(self):
        return self.target_model

    def getDataset(self):
        return self.dataset_name

    def prepareAttackData(self):

        # Shaping attack training data
        self.X_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size)*self.n_shadow_models,1))

        self.y_train_att = np.zeros(((self.shadow_train_size + self.shadow_val_size)*self.n_shadow_models,1))
        self.y_true_att = np.zeros(((self.shadow_train_size + self.shadow_val_size)*self.n_shadow_models,))

        # Preparing attack validation
        X_train_target, y_train_target = prepare_target_data(self.target_train)
        X_val_target, y_val_target = prepare_target_data(self.target_val)

        # BlackBox prediction
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

    def trainAttack(self, shadow_data, shadow_train_size, n_shadow_models, shadow_params, attack_params):
        print("Training " + self.dataset_name + " attack")

        self.n_shadow_models = n_shadow_models
        self.shadow_data = shadow_data
        self.shadow_train_size = shadow_train_size
        # Check
        self.shadow_val_size = shadow_train_size
        self.shadow_params = shadow_params
        self.attack_params = attack_params

        self.prepareAttackData()

        self.trainShadowModels()





a = Attack('model', 'model_name', 'texas', 'bb')
a.trainAttack('shadow_data', 100, 10, 'target_params', 'attack_params')
