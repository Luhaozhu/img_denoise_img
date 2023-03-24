import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import kornia
from tqdm import tqdm
from utils import VGGPerceptualLoss
import argparse

from data_provider import Video_Provider
from net.edvc_s_v3 import Network

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
        shuffle=False,
        num_workers=0
    )
    return data_loader

def load_model(model_path):
    model = torch.load(model_path)
    return model

def loss_fn_kd(stu_output,teacher_output,gt,alpha,T,loss_type,device,loss_fn_perceptual):
    if loss_type == "l1":
        loss_fn = nn.L1Loss()
    elif loss_type == "ssim":
        loss_fn = kornia.losses.SSIMLoss(window_size=5)
    elif loss_type == "mixed":
        loss_fn = kornia.losses.MS_SSIMLoss()
    elif loss_type == "perceptual":
        loss_fn_l1 = nn.L1Loss()

        general_loss = loss_fn_l1(stu_output,gt)
        perceptual_loss = loss_fn_perceptual(stu_output,gt)
        soft_loss = loss_fn_perceptual(stu_output/T,teacher_output/T) * alpha + perceptual_loss  * (1 - alpha)

    
        return soft_loss,general_loss


    general_loss = loss_fn(stu_output,gt)
    soft_loss = loss_fn(stu_output/T,teacher_output/T)

    return general_loss * (1 - alpha) + soft_loss * alpha,general_loss


def fetch_teacher_outputs(teacher, train_loader,device):
    print('-------Fetch teacher outputs-------')
    teacher.eval().to(device)
    #list of tensors
    teacher_outputs = []
    with torch.no_grad():
        #trainloader gets bs images at a time. why does enumerate(tl) run for all images?
        for i, (img, gt) in enumerate(train_loader):
            for idx in range(30):
                ft1 = img[:,idx].to(device)  # the t-th input frame
                if idx == 0:
                    pre=ft1
                else:
                    pre = ft0_fusion_data
                refine_out = teacher(torch.cat([pre.float().to(device),ft1.float().to(device)],1)*256)
                ft0_fusion_data = refine_out / 256

                teacher_outputs.append(ft0_fusion_data)  # [0,1]区+间


    return teacher_outputs


def train_student(current_epoch,total_epoch,student,teacher_outputs,
                  optimizer,train_loader,device,alpha,T,loss_type,loss_fn_perceptual):
    print('-------Train student-------')
    #called once for each epoch
    student.train().to(device)

    total_kd_loss = 0
    total_general_loss = 0
    count = 0
    with tqdm(total=310, desc=f'Epoch {current_epoch + 1}/{total_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iter, (data, gts) in enumerate(train_loader):
            batch_kd_loss = 0
            batch_general_loss = 0 
            for idx in range(30):
                teacher_output = teacher_outputs[count]
                img = data[:,idx].to(device)
                gt = gts[:,idx].to(device)

                if idx == 0:
                    pre = img
                else:
                    pre = student_output

                input = torch.cat([pre.float(),img.float()],1) * 256
                input = torch.tensor(input,dtype=torch.float32)
                refine_out = student(input)
                student_output = refine_out / 256

                loss,general_loss = loss_fn_kd(student_output, teacher_output, gt, alpha, T,loss_type,device,loss_fn_perceptual)    

                batch_kd_loss += loss
                batch_general_loss += general_loss
                count += 1

    
            iter_kd_loss = batch_kd_loss / 30

            optimizer.zero_grad()
            iter_kd_loss.backward()
            nn.utils.clip_grad_norm_(parameters=student.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            iter_general_loss = batch_general_loss / 30

            total_kd_loss += iter_kd_loss.item()
            total_general_loss += iter_general_loss.item()

            avg_kd_loss = total_kd_loss / (iter + 1)
            avg_general_loss = total_general_loss / (iter + 1)

            pbar.set_postfix(**{'kd_loss': avg_kd_loss,
                        loss_type:avg_general_loss,
                        # 'ssim': ssim,
                        'lr': get_lr(optimizer)})
            pbar.update(1)

    return student,avg_general_loss




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
                    # pre = torch.clip(ft0_fusion_data*255,0,255) / 256
                refine_out = model(torch.cat([pre.float().to(device),ft1.float().to(device)],1)*256)
                loss_l1_Charbonnier = loss_fn(refine_out/256, fgt)
                ft0_fusion_data = refine_out/256

                total_loss += loss_l1_Charbonnier.detach().item()
                count += 1
    avg_loss = float(total_loss) / float(count)
    print('-------------success---------------------')
    print(f'avg loss = {avg_loss}')

    return avg_loss




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda:0",help="cuda:0,cuda:1,cpu")
    parser.add_argument("--data_path",type=str,default="./train/")
    parser.add_argument("--teacher_model_path",type=str,default="./model/16_Epoch5750-Total_Loss0.0084.pth")
    parser.add_argument("--student_model_path",type=str,default="./model/16_Epoch3300-Total_Loss0.0078.pth")
    parser.add_argument("--stu_model_name",type=str,default="edvc_36G")
    parser.add_argument("--epoch",type=int,default=10000,help="training epochs")
    parser.add_argument("--alpha",type=float,default=0.9,help="alpha for knowledge distillation")
    parser.add_argument("--T",type=int,default=1,help="distillation temperature")
    parser.add_argument("--loss_type",type=str,default="l1",help="l1,ssim,mixed,perceptual")

    args = parser.parse_args()

    # create log directory
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())   
    output_path_name = f"{time_now}_distillation_{args.stu_model_name}_alpha{args.alpha}_loss_{args.loss_type}"
    output_path = os.path.join("./output",output_path_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    device = torch.device(args.device)
    train_dataloader = load_data(args.data_path)
    teacher_model = load_model(args.teacher_model_path)
    student_model = load_model(args.student_model_path)
    # student_model = Network()
    
    # get teacher outputs
    teacher_outputs = fetch_teacher_outputs(teacher_model,train_dataloader,device)

    # ************training******************
    optimizer = optim.Adam(student_model.parameters(), lr=0.0001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.92)
    if args.loss_type == "perceptual":
        loss_fn_perceptual = VGGPerceptualLoss(device)
    else:
        loss_fn_perceptual = None
    for e in range(args.epoch):

        student_model,general_loss = train_student(current_epoch=e,total_epoch=args.epoch,
                                                    student=student_model,teacher_outputs=teacher_outputs,
                                                    optimizer=optimizer,train_loader=train_dataloader,
                                                    device=device,alpha=args.alpha,T=args.T,loss_type=args.loss_type,
                                                    loss_fn_perceptual=loss_fn_perceptual)
        
        if e%50==0:
            save_path = os.path.join(output_path,f"Epoch{e}-Total_Loss{round(general_loss,5)}.pth")
            torch.save(student_model,save_path)

        lr_scheduler.step()
        


