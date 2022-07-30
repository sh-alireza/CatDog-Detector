import streamlit as st
import torch, torchvision
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor, Resize
import numpy as np

st.set_page_config(page_title="Page Title")
st.write('''
<style>
    p,h1,h2{
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)
st.write("<h2>Hi, Welcome to ...</h2>", unsafe_allow_html= True)
st.write("<h1>Cat or Dog?</h1>", unsafe_allow_html= True)
st.write("<br><br>", unsafe_allow_html= True)


col1, col2 = st.columns(2)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# model implementation

model = torch.jit.load("CatDogModel.pt")
model.eval()

transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize((500,500))
])

def inference(model, img, device="cpu"):

    with torch.no_grad():
        image = np.array(Image.open(img))
        
        image = transforms(image)
        
        image = image.unsqueeze(0)
        pred = model(image.to(device))
        return pred



# get the data

with col1.form("my_form", True):

    path = st.text_input("Link:")
    
    submit = st.form_submit_button("Predict")

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



img_file = col2.file_uploader("upload: ")

if img_file:

    pred = inference(model, img_file)
    pred_idx = np.argmax(pred)
    pred_label = "cat" if pred_idx == 0 else "dog"
    col2.image(img_file, width=100)
    col2.write(f"Predicted: {pred_label}, Prob: {pred[0][pred_idx]*100}%")
