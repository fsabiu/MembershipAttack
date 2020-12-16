from abc import ABC, abstractmethod

from lime.lime import lime_tabular
import numpy as np


class Explanator(ABC):
    def __init__(self, dataset, class_values, categorical_features, categorical_names, n_classes, black_box):
        # Data and target
        self.dataset = dataset
        self.class_values = class_values
        self.categorical_names = categorical_names
        self.categorical_features = categorical_features
        self.n_classes = n_classes
        self.black_box = black_box

        # Explainer
        self.explainer = None

    #@abstractmethod
    def explain(self):
        print("I'm abstract")

class LoreExplanator(Explanator):

    def __init__(self, dataset, class_values, categorical_features, categorical_names, n_classes, black_box):
        super().__init__(dataset, class_values, categorical_features, categorical_names, n_classes, black_box)


    def explain(self):
        print("I'm Lore")

class LimeExplanator(Explanator):
    def __init__(self, dataset, class_values, categorical_features, categorical_names, n_classes, black_box):
        super().__init__(dataset, class_values, categorical_features, categorical_names, n_classes, black_box)

        # Initializing Explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.dataset,
            class_names = self.class_values,
            feature_names = list(self.dataset.columns),
            categorical_features = self.categorical_features,
            categorical_names= self.categorical_names,
            kernel_width = 3,
            verbose = False)

        return self.explainer

    def generateNeighbors(self):
        print("I'm Lime")



"""black_box = load_obj(folder + 'target/RF/RF_model')

adult = 'mapped_dataset'
n_classes = 2
black_box = 'black_box'

e = LimeExplanator(dataset = 'adult',
                class_name = 'class',
                categorical_features = [],
                categorical_names = '100',
                n_classes = 2,
                black_box = 'black_box')

e.generateNeighbors()"""
