import torch, torchvision
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor, Resize
import numpy as np
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model implementation

model = torchvision.models.resnet18()
model.fc = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=2),
    nn.Softmax(dim=1)
])
model.load_state_dict(torch.load('ResNet_CatDog_v2.pth'))
model.eval()

# get the data

path = str(input("insert the image url: "))
# resp = requests.get(url)


transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize((500,500))
])

def inference(path, model, device="cpu"):
    try:
        resp = requests.get(path, timeout=10)
        print("request sent")
    except:
        return False
    
    with torch.no_grad():
        image = np.array(Image.open(BytesIO(resp.content)))
        
        image = transforms(image)
        
        image = image.unsqueeze(0)
        pred = model(image.to(device))
        return pred


pred = inference(path, model)
if torch.is_tensor(pred):
    pred_idx = np.argmax(pred)

    pred_label = "cat" if pred_idx == 0 else "dog"
    
    print(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx]*100}%")
else:
    print("can not get the url!!!")
