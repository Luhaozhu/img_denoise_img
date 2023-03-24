import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from typing import Tuple
import numpy as np
from PIL import Image
import tqdm
import cv2
from scipy import signal

class GaussianBlur(object):
    def __init__(self, kernel_size=7, sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.gaussian_kernel()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size), dtype=np.float)
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):  # [-r, r]
            for x in range(-radius, radius + 1):
                # 二维高斯函数
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v  # 高斯函数的x和y值 vs 高斯核的下标值
        kernel2 = kernel / np.sum(kernel)
        return kernel2

    def filter(self, img):
        img_arr = np.array(img)
        if len(img_arr.shape) == 2:
            new_arr = signal.convolve2d(img_arr, self.kernel, mode="same", boundary="symm")
        else:
            h, w, c = img_arr.shape
            new_arr = np.zeros(shape=(h, w, c), dtype=np.float)
            for i in range(c):
                new_arr[..., i] = signal.convolve2d(img_arr[..., i], self.kernel, mode="same", boundary="symm")
        new_arr = np.array(new_arr)
        return new_arr.transpose(2,0,1)

class Video_Provider(Dataset):
    def __init__(self, base_path,txt_file):
        super(Video_Provider, self).__init__()
        self.base_path = base_path
        # self.flow_path=flow_path
        self.txt_file = txt_file
        self.gt=[]
        self.blur = []
        self.flow=[]
        self.guss=GaussianBlur()
        self.KB=[77.50141960574743,16278.976637616463]
        for file in tqdm.tqdm(self.txt_file, desc="正在载入raw文件至内存"):
            data = np.load(self.base_path+file.strip('\n'))
            # flow=np.load(self.flow_path+file.strip('\n'))
            # im = torch.from_numpy(data).permute(0,3,1,2).to(self.device)
            data=data.reshape(30,256,256,1)
            im = data.transpose(0, 3, 1, 2)
            self.gt.append(im)
            # data=data.reshape(30,256,256)
            # im = data.transpose(1, 2,0)
            # blur=self.guss.filter(im)
            # blur=blur.reshape(30,256,256,1).transpose(0, 3, 1, 2)
            # # cv2.imwrite('./blur/' + file.strip('\n').split('.')[0] + '.bmp', np.array(blur[0][0], dtype=np.uint8))
            # self.blur.append(blur)
            # self.flow.append((flow))
    def __getitem__(self, index):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num=np.random.randint(1,64)
        K = torch.tensor(self.KB[0]/65535, device=self.device).reshape(-1,1, 1)
        B = torch.tensor((self.KB[1]/65535**2), device=self.device).reshape(-1,1, 1)
        im = torch.from_numpy(self.gt[index]/255/num).to(self.device)
        sigma = K * im + B
        std = torch.sqrt(torch.clamp_min(sigma, 0))
        noise = torch.normal(0, std)
        noised_img = torch.clamp(im + noise, 0.0, 1.0)
        # gt=torch.from_numpy(self.gt[index]/255/num).to(self.device)
        return noised_img*num, im*num

    def __len__(self):
        return len(self.gt)
class KSigma:
    def __init__(self, K_coeff: Tuple[float, float], B_coeff: Tuple[float, float], anchor: float):
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
    def tran(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)
        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a
        img = img_01
        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k
        return img,cvt_k,cvt_b
class Video_Provider_1(Dataset):
    def __init__(self, base_path, txt_file):
        super(Video_Provider_1, self).__init__()
        self.base_path = base_path
        self.txt_file = txt_file
        self.gt=[]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.KB = [77.50141960574743, 16278.976637616463]
        for file in tqdm.tqdm(self.txt_file, desc="正在载入raw文件至内存"):
            data = np.load(self.base_path+file.strip('\n'))
            data=data.reshape(30,256,256,1)
            im = torch.from_numpy(data[0:30:14]).permute(0,3,1,2).to(self.device)
            try:
                self.gt = torch.cat((self.gt,im), 0)
            except:
                self.gt=im
    def __getitem__(self, index):
        num = np.random.randint(1, 9)
        K = torch.tensor(self.KB[0] / 65535, device=self.device).reshape(-1, 1, 1)
        B = torch.tensor((self.KB[1] / 65535 ** 2), device=self.device).reshape(-1, 1, 1)
        im = self.gt[index] / 255 / num
        sigma = K * im + B
        std = torch.sqrt(torch.clamp_min(sigma, 0))
        noise = torch.normal(0, std)
        noised_img = torch.clamp(im + noise, 0.0, 1.0)
        # im_8=torch.nn.functional.interpolate(im, scale_factor=1 / 8)
        return noised_img, im

    def __len__(self):
        return self.gt.shape[0]
class Video_Provider_2(Dataset):
    def __init__(self, base_path, txt_file):
        super(Video_Provider_2, self).__init__()
        self.base_path = base_path
        self.txt_file = txt_file
        self.gt=[]
        self.blur = []
        self.flow=[]
        self.guss=GaussianBlur()
        self.KB=[77.50141960574743,16278.976637616463]
        for file in tqdm.tqdm(self.txt_file, desc="正在载入raw文件至内存"):
            data = np.load(self.base_path+file.strip('\n'))
            # flow=np.load(self.flow_path+file.strip('\n'))
            # im = torch.from_numpy(data).permute(0,3,1,2).to(self.device)
            data=data.reshape(30,256,256)
            # im = data.transpose(0, 3, 1, 2)
            for num in range(3):
                self.gt.append(data[num*14:num*14+2,:,:])
    def __getitem__(self, index):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num=np.random.randint(1,17)
        K = torch.tensor(self.KB[0]/65535, device=self.device).reshape(-1,1, 1)
        B = torch.tensor((self.KB[1]/65535**2), device=self.device).reshape(-1,1, 1)
        im = torch.from_numpy(self.gt[index]/255/num).to(self.device)
        sigma = K * im + B
        std = torch.sqrt(torch.clamp_min(sigma, 0))
        noise = torch.normal(0, std)
        noised_img = torch.clamp(im + noise, 0.0, 1.0)
        # gt=torch.from_numpy(self.gt[index]/255/num).to(self.device)
        return noised_img, im

    def __len__(self):
        return len(self.gt)
if __name__ == '__main__':
    path = './train/'
    # path.s
    files = os.listdir(path)
    data_set = Video_Provider(
            base_path=path,
            txt_file=files,
        )
    tran = transforms.ToPILImage()
    for index, (data, gt) in enumerate(data_set):
        # for i in range(6):
        #     tran(data[i, ...]).save('{}_noisy_{}.png'.format(index, i), quality=100)
        # tran(gt).save('{}_gt.png'.format(index), quality=100)
        print(data.shape)