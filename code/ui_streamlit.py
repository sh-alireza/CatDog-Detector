import streamlit as st
import torch, torchvision
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor, Resize
import numpy as np
import torch.nn as nn

st.set_page_config(page_title="Page Title")

st.subheader("Hi, Welcome to ...")
st.title("Cat or Dog?")
st.write("This is the passage?")
st.write("[< Don't click this link > ](https://google.com)")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model implementation

model = torchvision.models.resnet18()
model.fc = nn.Sequential(*[
    nn.Linear(in_features=512, out_features=2),
    nn.Softmax(dim=1)
])
model.load_state_dict(torch.load('ResNet_CatDog_v2.pth'))
model.eval()

transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize((500,500))
])

def inference(model, img, device="cpu"):

    with torch.no_grad():
        image = np.array(Image.open(img))
        
        image = transforms(image)
        # image = np.expand_dims(image, 1)
        image = image.unsqueeze(0)
        pred = model(image.to(device))
        return pred



# get the data
with st.form("my_form"):

    path = st.text_input("Link:")
    
    submit = st.form_submit_button("Click")
    if submit:

        try:
            resp = requests.get(path, timeout=10)
            print("request sent")
            img_link = BytesIO(resp.content)
        except:
            img_link = False

        if path != "":
            pred = inference(model, img_link)

            if torch.is_tensor(pred):
                pred_idx = np.argmax(pred)

                pred_label = "cat" if pred_idx == 0 else "dog"
                st.image(path,width=100)
                st.write(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx]*100}%")
                st.balloons()
            else:
                st.write("can not get the url!!!")


img_file = st.file_uploader("upload: ")

if img_file:

    pred = inference(model, img_file)
    pred_idx = np.argmax(pred)
    pred_label = "cat" if pred_idx == 0 else "dog"
    st.image(img_file, width=100)
    st.write(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx]*100}%")
