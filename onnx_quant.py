import onnxruntime
from onnxruntime.quantization import QuantType,quantize_dynamic,CalibrationDataReader,quantize_static
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import cv2
import numpy as np
from data_provider import Video_Provider


def load_data(data_path='./noise_9/',fpn_path='avg_2.tif'):
    # TODO(Aaron)：把数据导入的方法写好
    all_files = os.listdir(data_path)

    fpn = cv2.imread(fpn_path,-1)
    data_loader = []
    for i in range(len(all_files)):
        print(i)
        img_path = os.path.join(data_path,f'{str(i)}.tif')
        # data=cv2.imread(path+str(i)+'.tif',-1)
        data = cv2.imread(img_path,-1)
        if data is None:
            raise "Image does not exist, please check image path"
        data=np.clip((data-fpn)/65535*64,0,1)  
        data=torch.from_numpy(data.reshape(1,1,1024,1280))
        data_loader.append(data)
    
    return data_loader

def load_model(model_path,device='cpu'):
    model = torch.load(model_path).to(device)
    return model

def batch_reader(torch_dataloader,model_fp32_torch):
    model_fp32 = load_model(model_fp32_torch)
    model_fp32.eval()
    for iter, data in tqdm(enumerate(torch_dataloader)):
        ft1 = data  # the t-th input frame
        if iter == 0:
            pre=ft1
        else:
            pre = ft0_fusion_data
        input_data = torch.cat([pre.float(),ft1.float()],1)*256
        ft0_fusion_data = model_fp32(input_data.cpu())
        ft0_fusion_data = ft0_fusion_data / 256.
        yield {"input":input_data.cpu().detach().numpy()}
        
class DataReader(CalibrationDataReader):
    def __init__(self,torch_dataloader,model_fp32_torch):
        self.datas = batch_reader(torch_dataloader,model_fp32_torch)
 
    def get_next(self):
        return next(self.datas, None)


def dynamic_ptq(model_fp32,model_quant_dynamic):
    # 动态量化
    quantize_dynamic(
        model_input=model_fp32, # 输入模型
        model_output=model_quant_dynamic, # 输出模型
        weight_type=QuantType.QInt8, # 参数类型 Int8 / UInt8
        optimize_model=True # 是否优化模型
    )

def static_ptq(model_fp32,model_fp32_torch,model_quant_static,torch_dataloader):
    data_reader = DataReader(torch_dataloader,model_fp32_torch)

    # 静态量化
    quantize_static(
        model_input=model_fp32, # 输入模型
        model_output=model_quant_static, # 输出模型
        calibration_data_reader=data_reader, # 校准数据读取器
        # quant_format= QuantFormat.QDQ, # 量化格式 QDQ / QOperator
        activation_type=QuantType.QUInt8, # 激活类型 Int8 / UInt8
        weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
        # calibrate_method=CalibrationMethod.MinMax, # 数据校准方法 MinMax / Entropy / Percentile
        # optimize_model=True # 是否优化模型
    )

def quantize(static=True):
    model_fp32 = "./model/edvc_s_v2.onnx"
    model_quant_dynamic = "./model/edvc_int8_dynamic.onnx"
    model_fp32_torch = "./model/16_Epoch5750-Total_Loss0.0084.pth"
    model_quant_static = "./model/edvc_int8_onnx_static_version8.onnx"
    if not static:
        dynamic_ptq(model_fp32,model_quant_dynamic)
    else:
        torch_dataloader = load_data()
        static_ptq(model_fp32,model_fp32_torch,model_quant_static,torch_dataloader)

if __name__ == "__main__":

    static = True
    quantize(static)
    
