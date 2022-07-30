import torch

# model implementation

model = torch.jit.load("CatDogModel.pt")

dummy_input = torch.randn(1, 3, 500, 500, requires_grad=True)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"]
)
