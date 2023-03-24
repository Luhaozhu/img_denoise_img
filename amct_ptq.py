import os
import amct_pytorch 
from amct_pytorch.accuracy_based_auto_calibration import AutoCalibrationEvaluatorBase
from typing import Tuple
import torch.nn as nn
import torch
from data_provider import Video_Provider
from torch.utils.data import DataLoader
import numpy as np
import argparse

def load_data(data_path='./train/'):
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

def model_forward(model,data_loader,device):
    model.eval()
    total_loss = 0
    count = 0
    loss_fn = nn.L1Loss()
    with torch.no_grad():
        for iter, (data, gt) in enumerate(data_loader):
            for idx in range(30):
                ft1 = data[:,idx].to(device)  # the t-th input frame
                fgt = gt[:,idx].to(device)  # the t-th gt frame
                if idx == 0:
                    pre=ft1
                else:
                    pre = ft0_fusion_data
                refine_out = model(torch.cat([pre.float().to(device),ft1.float().to(device)],1)*256)
                loss_l1_Charbonnier = loss_fn(refine_out/256, fgt)
                ft0_fusion_data = refine_out/256

                total_loss += loss_l1_Charbonnier
                count += 1
    avg_loss = float(total_loss) / float(count)
    print('-------------success---------------------')
    print(f'avg loss = {avg_loss}')
    return avg_loss

class ModelEvaluator(AutoCalibrationEvaluatorBase):
    def __init__(self,data_loader,diff,device):
        super().__init__()
        self.loss = nn.L1Loss()
        self.dataloader = data_loader
        self.device = device
        self.diff = diff  # 0.0084   0.0005

    def calibration(self, model):
        model_forward(model,self.dataloader,self.device)
            
    def evaluate(self, model):
        loss = model_forward(model,self.dataloader,self.device)
        return loss
        # if torch.cuda.is_available():
            # torch.cuda.empty_cache()
    
    def metric_eval(self, original_metric, new_metric) -> Tuple[bool, float]:
        loss_diff = original_metric - new_metric
        if loss_diff < self.diff:
            return True,loss_diff
        else:
            return False,loss_diff
        
def main():
    # Step 1: load data and model
    calib_dataloader = load_data(data_path="./train/")
    model_fp32_to_quantize = load_model(model_path="model/16_Epoch5750-Total_Loss0.0084.pth")

    device = torch.device("cpu")
    model_fp32_to_quantize.eval()
    model_fp32_to_quantize.to(device)
    # export_onnx_file = 'edvc_s_v6.onnx'
    # dummy_input = torch.randn([1,2,1024,1280])
    # torch_out = torch.onnx.export(model_fp32_to_quantize,dummy_input,export_onnx_file,
    #                 input_names=['input'],output_names=['output'],
    #                 dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})

    # Step2: create quant config json file
    skip_layers = ["enc3.3.conv1.conv", "dec4.upsample", "dec2.proj_conv.conv"]
    batch_num = 8
    diff = 0.0003
    input_data = torch.randn([1,2,1024,1280])
    output_path_name = f"edvc_batch{batch_num}_diff{diff}_skip{len(skip_layers)}"
    output_path = os.path.join("output",output_path_name)
    if os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path,"config"))
    # config_json_file = os.path.join("config","config_edvc.json")
    config_json_file = os.path.join(os.path.join(output_path,"config"),"config_edvc.json")

    amct_pytorch.create_quant_config(config_json_file,model_fp32_to_quantize,
                                     input_data,skip_layers,batch_num)
    
    # step3: construct the instance of AutoCalibration Evaluator
    
    evaluator = ModelEvaluator(calib_dataloader,diff=diff,device=device)

    # step4: quantize the model
    record_file = os.path.join(output_path,"scale_offset_record.txt")
    result_path = os.path.join(output_path,"model")
    amct_pytorch.accuracy_based_auto_calibration(
        model=model_fp32_to_quantize,
        model_evaluator=evaluator,
        config_file=config_json_file,
        record_file=record_file,
        save_dir=result_path,
        input_data=input_data,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':{0:'batch_size'},
            'output':{0:'batch_size'}
        },
        strategy='BinarySearch',
        sensitivity='CosineSimilarity'
    )

if __name__ == "__main__":

    main()
