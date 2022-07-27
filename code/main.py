import torch, torchvision
from torchvision.transforms import ToTensor, Resize
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CatDogDataset
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import copy

# Constants

TRAIN_PATH = "../data/train"
VAL_PATH = "../data/val"
NUM_BATCH = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformers

transform = transforms.Compose([
    ToTensor(),
    Resize((500,500))
])

# dataset & dataloader

train_data = CatDogDataset(TRAIN_PATH, transform)
val_data = CatDogDataset(VAL_PATH, transform)


train_dl = DataLoader(train_data, batch_size=NUM_BATCH)
val_dl = DataLoader(val_data, batch_size=NUM_BATCH)

# import pretrained model

model = torchvision.models.resnet18(pretrained=True)

# freeze weights

for param in model.parameters():
    param.requires_grad = False

# finetune the last fully connected layer to prefered output (2)

model.fc = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=2),
    nn.Softmax(dim=1)
])

# validation and train functions

def validate(model, data):
    total = 0
    correct = 0

    for (images, labels) in data:
        images = images.to(DEVICE)
        x = model(images)
        _, pred = torch.max(x, 1)
        
        total += x.size(0)
        correct += torch.sum(pred == labels)
        
        
    return correct*100/total


def train(num_epoch = EPOCHS, lr = LEARNING_RATE, device = DEVICE):
    accuracies = []
    cnn = model.to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)

    max_accuracy = 0

    for epoch in range(num_epoch):
        for i, (images, labels) in tqdm(enumerate(train_dl)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn,val_dl))
        accuracies.append(accuracy)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("saving best model with accuracy: ", accuracy)
        print("Epoch: ", epoch+1, "Accuracy: ", accuracy, "%")

    # plt.plot(accuracies)
    return best_model


# train model with own dataset

resnet = train()

# save the best model

torch.save(resnet.state_dict(), "ResNet_CatDog_v2.pth")
