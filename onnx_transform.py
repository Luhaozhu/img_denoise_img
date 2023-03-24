import torch

def onnx_export(model,dummy_input,export_onnx_file):
    torch.onnx.export(model,dummy_input,export_onnx_file,
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})

if __name__ == "__main__":
    export_onnx_file = './model/edvc_512.onnx'
    dummy_input = torch.randn(1,2,512,512)
    model = torch.load("/data/aaron/quantization_deploy/img_denoise/model/16_Epoch5750-Total_Loss0.0084.pth").to("cpu")
    # model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
    onnx_export(model,dummy_input,export_onnx_file)