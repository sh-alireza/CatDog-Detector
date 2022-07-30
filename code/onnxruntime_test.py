import torch, torchvision
import requests
from PIL import Image
from io import BytesIO
from torchvision.transforms import ToTensor, Resize
import numpy as np
import onnx
import onnxruntime


model = torch.jit.load("CatDogModel.pt")

dummy_input = torch.randn(1, 3, 500, 500, requires_grad=True)
torch_out = model(dummy_input)

transforms = torchvision.transforms.Compose([
    ToTensor(),
    Resize((500,500))
])
#check the mdoel
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# transfer to onnx runtime
ort_session = onnxruntime.InferenceSession("model.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
ort_outs = ort_session.run(None, ort_inputs)

path = str(input("insert the image url: "))
resp = requests.get(path, timeout=10)
image = np.array(Image.open(BytesIO(resp.content)))
image = transforms(image)
image = image.unsqueeze(0)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]
print(img_out_y)
