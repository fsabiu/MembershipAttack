class Attack(object):

    def __init__(self, target, targetModel, dataset):
        self.target = target
        self.targetModel = targetModel
        self.dataset = dataset


    def getTarget(self):
        return self.target
