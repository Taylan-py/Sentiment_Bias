# What should this script do?
# -> Load model
# -> Evaluate, plot on graph

# Imports
from start import * # imports hyperparameters aswell, set gpu variable accordingly
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Relative Path to model
    #model_name = "./output/converted_model_1000_epochs_01_lr_136_batch"
    model_name = "./output/normal_model_1000_epochs_01_lr_136_batch"


    # Initialize Dataset
    test_sets = TheDataset(convert_output, "test")
    test_loaders = DataLoader(test_sets, 1, shuffle=True)

    # initialize model
    if gpu:
        model = nn.Sequential(nn.Linear(test_set.get_n_input(), n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_out),
                    nn.Sigmoid()).to('cuda')
    else:
        model = nn.Sequential(nn.Linear(test_sets.get_n_input(), n_hidden),
                    nn.ReLU(),
                    nn.Linear(n_hidden, n_out),
                    nn.Sigmoid())

    # Load model based on model_name variable
    model.load_state_dict(torch.load(model_name))
    model.eval()

    a, b = test(model, test_loaders)

    figure, axis = plt.subplots(2, 2)

    f1 = plt.figure(3)
    plt.hist(a, bins = 10, range = (0, 1))
    plt.savefig('./output/'+model_name.split("/")[-1])

    f2 = plt.figure(2)
    plt.hist(b, bins = 10, range = (0, 1))
    plt.savefig('./output/'+model_name.split("/")[-1]+"groundtruth")


# check why conversion doesn't work
