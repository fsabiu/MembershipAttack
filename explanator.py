from abc import ABC, abstractmethod
import numpy as np

class Explanator(ABC):
    def __init__(self, dataset, class_name, n_classes, black_box):
        self.dataset = dataset
        self.class_name = class_name
        self.n_classes = n_classes
        self.black_box = black_box

    @abstractmethod
    def explain(self):
        print("I'm abstract")

class LoreExplanator(Explanator):

    def __init__(self, dataset, class_name, n_classes, black_box):
        super().__init__(dataset, class_name, n_classes, black_box)

    def explain(self):
        print("I'm Lore")

class LimeExplanator(Explanator):
    def __init__(self, dataset, class_name, n_classes, black_box):
        super().__init__(dataset, class_name, n_classes, black_box)

    def explain(self):
        print("I'm Lime")


e = LimeExplanator('texas', 'class', '100', 'bb')
e.explain()
