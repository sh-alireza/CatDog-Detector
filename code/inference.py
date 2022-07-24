import torch, torchvision
import torch.nn.functional as F
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor, Resize
import numpy as np
import torch.nn as nn

DEVICE = "cpu"

# model implementation

model = torchvision.models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=2)
model.load_state_dict(torch.load('ResNet_CatDog.pth'))
model.eval()

# get the data

path = str(input("insert the image url: "))
# resp = requests.get(url)


transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize((50,50))
])

def inference(path, model, device="cpu"):
    resp = requests.get(path)
    print("request sent")
    
    with torch.no_grad():
        image = np.array(Image.open(BytesIO(resp.content)))
        
        image = transforms(image)
        # image = np.expand_dims(image, 1)
        image = image.unsqueeze(0)
        pred = model(image.to(device))
        return pred



pred = inference(path, model)
pred_idx = np.argmax(pred)

pred_label = "cat" if pred_idx == 0 else "dog"

print(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx]*100}%")

