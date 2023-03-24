import copy
import os
import torch
import torchvision
import torch.nn.utils.prune as prune
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx,convert_fx
from torch.ao.quantization import QConfigMapping
from thop import profile
import pandas as pd
import numpy as np
import math
from PIL import Image
import cv2
from tqdm import tqdm

from data_provider import Video_Provider

def load_data(data_path='./train/',use_random_set=False,batch_size=16):
    # TODO(Aaron)：把数据导入的方法写好
    files=os.listdir(data_path)
    data_set = Video_Provider(
        base_path=data_path,
        txt_file=files,
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    return data_loader

def load_model(model_path,device='cpu'):
    model = torch.load(model_path).to(device)
    return model

def evaluate_sparsity(model):
    """统计模型剪枝后的稀疏程度"""
    a,b = 0.,0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / (a+1e-5)


def prune_model(model_fp32_to_prune,amount_conv=0.3,amount_linear=0.5):
    """
    prune fp32 model to be more sparse
    model_fp32_to_prune(torch.nn.Module): model that prepared to be pruned
    amount_conv(float): convolution layer prune ratio
    amount_linear(float): fully connected layer prune ratio
    """
    for name,module in model_fp32_to_prune.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            # 对卷积层的权重和偏置量进行修剪
            prune.l1_unstructured(module,name='weight',amount=amount_conv)
            # prune.l1_unstructured(module,name='bias',amount=amount_conv)
            # 将name保持永久修剪，同时去除参数的前向传播钩子
            prune.remove(module,'weight')
        elif isinstance(module,torch.nn.Linear):
            prune.l1_unstructured(module,name='weight',amount=amount_linear)
            # prune.l1_unstructured(module,name='bias',amount=amount_conv)
            prune.remove(module,'weight')

    return model_fp32_to_prune
    

def calibrate(model,data_loader,device='cpu'):
    """
    model: ObservedGraphModule 输入的量化模型
    calib_dataloader: 校准数据集
    """
    model.eval()
    with torch.inference_mode():
        for iter, (data, gt) in tqdm(enumerate(data_loader)):
            for idx in range(30):
                ft1 = data[:,idx]  # the t-th input frame
                fgt = gt[:,idx]  # the t-th gt frame
                if idx == 0:
                    pre=ft1
                else:
                    pre = ft0_fusion_data
                input_data = torch.cat([pre.float().to(device),ft1.float().to(device)],1)*256
                ft0_fusion_data = model(input_data)
                ft0_fusion_data = ft0_fusion_data / 256.
    print('calibration is done.')


def quantize(model_fp32,dataloader):
    model_fp32.eval()
    qconfig = get_default_qconfig('fbgemm')
    # qconfig = get_default_qconfig('qnnpack')
    # 设置量化规则，convtranspose 2D activation由per tensor量化，weights由per channel量化
    # qconfig_trans2d = torch.quantization.qconfig.QConfig(
    #     activation = torch.quantization.observer.NoopObserver,
    #     # weight=torch.quantization.observer.default_per_channel_weight_observer
    #     weight=torch.quantization.observer.NoopObserver)

    qconfig_mapping = QConfigMapping().set_global(qconfig)
    qconfig_mapping = qconfig_mapping.set_object_type(torch.nn.ConvTranspose2d,None)
    prepared_model = prepare_fx(model_fp32,qconfig_mapping,example_inputs = (torch.randn(1, 2, 256, 256),))
    calibrate(prepared_model,dataloader)
    quantized_model = convert_fx(prepared_model)
    print(quantized_model)
    return quantized_model


def evaluate_model(model_fp32,model_fp32_prune,model_int8,val_loader,device='cpu'):

    # 1. evaluate parameters and TFLOPS
    input = torch.randn(1,3,224,224).to(device)
    flops_fp32,params_fp32 = profile(model_fp32,inputs=(input,))
    flops_fp32_prune,params_fp32_prune = profile(model_fp32_prune,inputs=(input,))
    flops_int8,params_int8 = profile(model_int8,inputs=(input,))
    

    # 2. evaluate model accuracy
    model_int8.eval()
    model_fp32.eval()
    model_fp32_prune.eval()

    corrects_fp32 = 0
    corrects_int8 = 0
    corrects_fp32_prune = 0
    dataset_length = len(val_loader) * 16
    with torch.inference_mode():
        for image,label in val_loader:
            predict_fp32 = model_fp32(image)
            predict_int8 = model_int8(image)
            predict_fp32_prune = model_fp32_prune(image)
            _,fp32_preds = torch.max(predict_fp32,1)
            _,fp32_prune_preds = torch.max(predict_fp32_prune,1)
            _,int8_preds = torch.max(predict_int8,1)
            corrects_fp32 += torch.sum(fp32_preds == label.data)
            corrects_fp32_prune += torch.sum(fp32_prune_preds == label.data)
            corrects_int8 += torch.sum(int8_preds == label.data)
        
        accuracy_fp32 = float(corrects_fp32) / dataset_length
        accuracy_fp32_prune = float(corrects_fp32_prune) / dataset_length
        accuracy_int8 = float(corrects_int8) / dataset_length
    
    # evaluate model sparsity
    fp32_sparsity = evaluate_sparsity(model_fp32)
    fp32_prune_sparsity = evaluate_sparsity(model_fp32_prune)
    int8_sparsity = evaluate_sparsity(model_int8)

    reports = pd.DataFrame({'model':['fp32','fp32_prune','int8'],
                            'TFLOPS':[flops_fp32,flops_fp32_prune,flops_int8],
                            'Params':[params_fp32,params_fp32_prune,params_int8],
                            'Accuracy':[accuracy_fp32,accuracy_fp32_prune,accuracy_int8],
                            'Sparsity':[fp32_sparsity,fp32_prune_sparsity,int8_sparsity]})
    reports.to_csv('output/torchfx_quantize_reports.csv')

if __name__ == "__main__":
    # Step 1: load data and model
    calib_dataloader = load_data(data_path="./train/")
    model_fp32_to_quantize = load_model(model_path="./model/16_Epoch5750-Total_Loss0.0084.pth")


    # Step 2: prune the model and evaluate accuracy
    # model_fp32_to_prune = copy.deepcopy(model_fp32)
    # model_fp32_to_quantize = prune_model(model_fp32_to_prune)

    # Step 3: quantize the model
    quantized_model_int8 = quantize(model_fp32_to_quantize,calib_dataloader)
    torch.jit.save(torch.jit.script(quantized_model_int8),'./model/edvc_int8.pth')

    # save as onnx format
    export_onnx_file = 'model/edvc_int8.onnx'
    dummy_input = torch.randn(1,2,1024,1280)
    torch.onnx.export(quantized_model_int8,dummy_input,export_onnx_file,
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}},
                      opset_version=13)
