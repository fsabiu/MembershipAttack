import sys

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
