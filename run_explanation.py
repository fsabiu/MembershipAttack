from explanator import *

def adult_explain(model_type, bb, r2E):

    n_classes = 2

    e = LimeExplanator(dataset = 'adult',
                    class_name = 'class',
                    categorical_features = [],
                    categorical_names = '100',
                    n_classes = 2,
                    black_box = 'black_box')

    

if __name__ == "__main__":
"""
Requires:
- data/dataset/results/attack_models/
to exists
"""
    if(len(sys.argv) != 3):
        print('Usage: ' + sys.argv[0] + ' model dataset')
        exit(1)

    # Target params
    model = sys.argv[1]
    dataset = sys.argv[2]

    if (model not in ['RF', 'NN']):
        print("Model not implemented")
        exit(1)

    if (dataset not in ['adult', 'mobility', 'texas']):
        print("Dataset not supported")
        exit(1)

    folder = 'data/' + dataset + '/'

    r2E = pd.read_csv(folder + 'baseline_split/r2E_mapped.csv', nrows = 1000)
    bb = None

    if (model == 'RF'):
        bb = load_obj(folder + 'target/' + model + '/RF_model')

    if(model == 'NN'):
        bb = load_obj(folder + 'target/' + model + '/RF_model')

    if(dataset == 'adult' and model == 'RF'):
        adult_explain(model, bb, R2E)

    if(dataset == 'mobility'):
        mobility_explain(model, bb, R2E)

    if(dataset == 'texas'):
        texas_explain(model, bb, R2E)
