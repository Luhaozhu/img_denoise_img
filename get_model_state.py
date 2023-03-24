import torch
from fvcore.nn import FlopCountAnalysis
import numpy as np
import torch.nn as nn
from distilation import load_data
# from net.edvc_s import Network
small_model = torch.load("/data/aaron/quantization_deploy/img_denoise/model/16_Epoch3300-Total_Loss0.0078.pth").cpu()
# big_model = torch.load("/data/aaron/quantization_deploy/img_denoise/model/16_Epoch5750-Total_Loss0.0084.pth").cpu()

# torch.save(small_model.cpu().state_dict(),"./model/edvc_36G_weights.pth")

# model = Network()
# ckpt = torch.load("./model/edvc_36G_weights.pth")

def model_evaluate(model,data_loader,device):
    model.eval().to(device)
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


# model.load_state_dict(ckpt)
dummy_input = torch.randn(1,2,1024,1280)
small_flops = FlopCountAnalysis(small_model,dummy_input)
# big_flops = FlopCountAnalysis(big_model,dummy_input)

print("FLOPs small:",small_flops.total() / 1e9)
# print("FLOPs big:",big_flops.total() / 1e9)

train_loader = load_data("./train/")
device = torch.device("cuda:1")

avg_loss = model_evaluate(small_model,train_loader,device)

