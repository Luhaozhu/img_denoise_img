import torch

# def load_model(model_path,device='cpu'):
#     model = torch.load(model_path)
#     return model

# model_fp32_to_quantize = load_model(model_path="unet.pth")
# export_onnx_file = 'unet.onnx'
# dummy_input = torch.randn([1,2,1024,1280])
# torch_out = torch.onnx.export(model_fp32_to_quantize,dummy_input,export_onnx_file,
#                 input_names=['input'],output_names=['output'],
#                 dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})

import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.autograd import Variable
from net.edvc_s import Network


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("/data/aaron/quantization_deploy/img_denoise/output/downsample_train_logs/16_Epoch100-Total_Loss0.0158.pth").cpu()

dummy_input = torch.randn(1, 8, 512, 640)
export_onnx_file = "edvc_downsample.onnx"
# model = Network().cpu().eval()
torch_out = torch.onnx.export(model,dummy_input,export_onnx_file,
            input_names=['input'],output_names=['output'],
            dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})