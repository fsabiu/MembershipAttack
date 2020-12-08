import numpy as np
import pandas as pd

def prepare_target_data(dataset, class_name):
    y = dataset[class_name].values
    dataset.drop([class_name], axis=1, inplace = True)

    cols = dataset.columns
    x = np.array(dataset[cols])

    return x, y
