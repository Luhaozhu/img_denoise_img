import os
from torch.utils.data import Dataset, DataLoader
from data_provider import Video_Provider
from net.edvc_s import *
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from net.edvc_s_downsample import Network,PixelUnShuffle
# import netloss
# from edvc_s_pruning1 import *
path='./train/'
files=os.listdir(path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def creat_gt(raw):
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[0::2, 1::2]
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[1::2, 0::2]
    bayer_RGGB=np.array([ch_R,ch_Gr,ch_B,ch_Gb])
    return bayer_RGGB


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def train():
    data_set = Video_Provider(
        base_path=path,
        txt_file=files,
    )
    data_loader = DataLoader(
        dataset=data_set,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    #
    model = Network()
    # model = model=torch.load('./logs/05_Epoch1550-Total_Loss0.0016.pth')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.92)
    if not os.path.exists('./output/downsample_train_logs_lr1e-4'):
        os.mkdir('./output/downsample_train_logs_lr1e-4')
    loss_func = nn.L1Loss()
    pixel_unshuffle = PixelUnShuffle(upscale_factor=2)
    Epoch=10000

    model.train()
    for e in range(0, Epoch):
        all_total_loss = 0
        total_l1_Charbonnierloss=0
        with tqdm(total=310, desc=f'Epoch {e + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iter, (data, gt) in enumerate(data_loader):
                
                loss_l1_Charbonnier_total=0
                for idx in range(30):
                    ft1 = data[:,idx].to(device)  # the t-th input frame
                    fgt = gt[:,idx].to(device)  # the t-th gt frame
                    if idx == 0:
                        pre=ft1
                    else:
                        pre = ft0_fusion_data

                    inp1 = pixel_unshuffle(pre.float())
                    inp2 = pixel_unshuffle(ft1.float())
                    refine_out = model(torch.cat([inp1,inp2],1))
                    ################################loss
                    loss_l1_Charbonnier = loss_func(refine_out, fgt)
                    loss_l1_Charbonnier_total += loss_l1_Charbonnier
                    # l1loss_total += loss_l1
                    ft0_fusion_data = refine_out
                total_loss = loss_l1_Charbonnier_total / 30#+wraploss_total*0.5/29#+perceptualLoss_total*0.1/30
                all_total_loss += total_loss.item()
                total_l1_Charbonnierloss += loss_l1_Charbonnier_total.item() / 30

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()

                pbar.set_postfix(**{'total_loss': all_total_loss/(iter+1),
                                    'l1loss':total_l1_Charbonnierloss/(iter+1),
                                    # 'ssim': ssim,
                                    'lr': get_lr(optimizer)})
                pbar.update(1)
        if e%50==0:
            torch.save(model,'./output/downsample_train_logs_lr2e-4/16_Epoch%d-Total_Loss%.4f.pth' % (
            (e), total_l1_Charbonnierloss/(iter+1)))
        lr_scheduler.step()


if __name__ == '__main__':
    train()
