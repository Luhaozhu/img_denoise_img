import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import amct_pytorch
import copy
import numpy as np

from data_provider import Video_Provider

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

def model_evaluate(model,data_loader,device):
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

                total_loss += loss_l1_Charbonnier.detach().item()
                count += 1
    avg_loss = float(total_loss) / float(count)
    print('-------------success---------------------')
    print(f'avg loss = {avg_loss}')
    return avg_loss

def train(model,train_dataloader,device,output_path,record_file,input_data):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # lr = 0.0001  # lr=5e-5
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.92)
    loss_func_l1 = nn.L1Loss()
    Epoch=200
    best_model_weights = copy.deepcopy(model.state_dict())

    model.train()
    for e in range(0, Epoch):
        model.to(device)
        all_total_loss = 0

        with tqdm(total=310, desc=f'Epoch {e + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iter, (data, gt) in enumerate(train_dataloader):
                loss_l1_Charbonnier_total=0
                for idx in range(30):
                    ft1 = data[:,idx].to(device)  # the t-th input frame
                    fgt = gt[:,idx].to(device)  # the t-th gt frame
                    if idx == 0:
                        pre = ft1
                    else:
                        pre = torch.clip(ft0_fusion_data*255,0,255) / 256
                    input = torch.cat([pre.float(),ft1.float()],1) * 256
                    input = torch.tensor(input,dtype=torch.float32)
                    refine_out = model(input)
                    ################################loss
                    loss_l1_Charbonnier=loss_func_l1(refine_out/256, fgt)
                    loss_l1_Charbonnier_total+=loss_l1_Charbonnier
                    ft0_fusion_data = refine_out/256

                total_loss =loss_l1_Charbonnier_total/30#+wraploss_total*0.5/29#+perceptualLoss_total*0.1/30
                all_total_loss+=total_loss.item()

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

                avg_loss = all_total_loss/(iter+1)
            
                pbar.set_postfix(**{'total_loss': avg_loss,
                        'l1loss':avg_loss,
                        # 'ssim': ssim,
                        'lr': get_lr(optimizer)})
                pbar.update(1)

        if e % 5 == 0:
            best_model_weights = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_model_weights)
            print(f"best model has been saved in epoch:{e},avg_loss = {avg_loss}")
            model_output_path = os.path.join(output_path,f"epoch{e}_loss{avg_loss}")
            if not os.path.exists(model_output_path):
                os.mkdir(model_output_path)
            model_evaluate(model,train_dataloader,device)
            amct_pytorch.save_compressed_retrain_model(model,record_file,model_output_path,input_data,
                                          input_names=['input'], output_names=['output'],
                                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            
        lr_scheduler.step()

    

    return model

    

def main(args):
    # Step0: set up device and load data
    train_dataloader = load_data(args.data_path)
    device = torch.device(args.device)

    # Step1: create model
    print("====> Create pretrained model edvc-------->")
    model = load_model(args.model_path,device)

    # Step2: Calculate origin model's loss
    avg_loss_orig = model_evaluate(model,train_dataloader,device)

    # Step3: Creating the retraining configuration file
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())   
    output_path_name = f"{time_now}_compressedretrain_{args.model_name}_uint8_lr5e-5"
    output_path = os.path.join("./output",output_path_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path,"config"))
    # config_file = os.path.join(os.path.join(output_path,"config"),"config_edvc.json")
    simple_cfg = None
    record_file = os.path.join(output_path,"record.txt")
    input_data = torch.randn([1,2,512,1280])
    
    # prepare input data with label
    img_data = []
    label_data = []
    img_num = 2
    
    
    # Step4 generate the retraining model in default graph and 
    # create the quantization factor record file
    print("=======> Create quant_retrain_model")
    model = amct_pytorch.create_compressed_retrain_model(model,input_data,args.config_definition,record_file)

    # Step5 retraining quantitative model and inferencing
    best_model = train(model,train_dataloader,device,output_path,record_file,input_data)

    # Step6 calculate quantized loss
    avg_loss_quantized = model_evaluate(best_model,train_dataloader,device)
    print(f"original loss:{avg_loss_orig}\n quantized loss:{avg_loss_quantized}")

    # Step7 save as onnx model

    print("========> save quant_retrain_model")
    best_model.to("cpu")
    amct_pytorch.save_compressed_retrain_model(best_model,record_file,output_path,input_data,
                                          input_names=['input'], output_names=['output'],
                                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    print("=============>success!!!<================")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:1",help="cuda:0,cuda:1,cpu")
    parser.add_argument("--data_path",type=str,default="./train/")
    parser.add_argument("--model_name",type=str,default="edvc_s")
    parser.add_argument("--model_path",type=str,default="./model/16_Epoch300-Total_Loss0.0086.pth")
    parser.add_argument('--config_definition', dest='config_definition', 
                        default=None, type=str, help='The simple configure define file.')
    args = parser.parse_args()

    main(args)