import sys
from util import load_obj
import tensorflow as tf

if __name__ == "__main__":
    """
    Requires:
    - data/dataset/results/attack_models/
    to exists
    """
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' modelRF modelBB dataset')
        exit(1)
    # Target params
    attackModelType = sys.argv[1]
    bbType = sys.argv[2]
    dataset_name = sys.argv[3]

    if (bbType not in ['RF', 'NN'] or attackModelType not in ['RF', 'NN'] ):
        print("Model not implemented")
        exit(1)

    if (dataset not in ['adult', 'mobility', 'texas']):
        print("Dataset not supported")
        exit(1)


    # Reading explained records
    explainedData = None
    explainedData = pd.read_csv('data/' + dataset_name + '/baseline_split/' + dataset_name +'r2E_mapped_explained.csv', dtype = dtypes)

    # Reading Black Box
    bb_folder = 'data/' + dataset_name + '/target/' + bbType + '/'
    bb = None
    if (bbType == "RF"):
        bb = load_obj(folder + '/RF_model')
    if(bbType == 'NN'):
        bb = keras.models.load_model('NN_model.h5')

    # Reading attack models
