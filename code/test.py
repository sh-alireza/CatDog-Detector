from re import sub
import torchvision
import os
import glob
# model = torchvision.models.resnet18(pretrained=True)
# print(model)
import subprocess

path = "/home/alireza/Deep/project-1/data/val"

cat_path = glob.glob("/home/alireza/Deep/project-1/data/train/cat*.jpg")
dog_path = glob.glob("/home/alireza/Deep/project-1/data/train/dog*.jpg")


for i in range (2499):
    subprocess.run(["mv", dog_path[i], path])

