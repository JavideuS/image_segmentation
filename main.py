from training import *
from loss import *
import training_data_loading
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

    optimizer = optim.Adam(model.parameters(), lr=0.11)
    loss = ModifiedJaccardLoss()
    li,le = train(model,loss,optimizer,num_epochs= 50, batch_size= 2)


if __name__ == '__main__':
    main()


