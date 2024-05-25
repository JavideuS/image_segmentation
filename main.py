from training import *
from loss import *
from model_zoo import *
import torch


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = ZeroNeurons()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.5)
    loss = ModifiedJaccardLoss()
    li , le = train(model, loss, optimizer, num_epochs = 10 , batch_size = 5)


if __name__ == '__main__':
    main()
