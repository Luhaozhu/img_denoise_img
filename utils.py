import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, device,resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].to(device).eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].to(device).eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].to(device).eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].to(device).eval())
        

        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x.float())
            y = block(y.float())
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        l1_loss = nn.L1Loss()
        total_loss = loss/len(self.blocks) * 0.5 + l1_loss(input,target) * 0.5
        return total_loss
    

if __name__ == "__main__":

    device = torch.device("cuda:0")
    input = torch.randn(1,1,256,256).to(device)
    target = torch.randn(1,1,256,256).to(device)
    perceptual_loss = VGGPerceptualLoss(device)

    loss = perceptual_loss(input,target)