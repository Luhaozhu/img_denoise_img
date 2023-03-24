import onnxruntime
import numpy as np
import math
import os
import cv2
import torch
from net.edvc_s_downsample import PixelUnShuffle
import time

def calculate_psnr(img1,img2):
    mse = np.mean((np.clip(img1.transpose(1,2,0),0,1) - np.clip(img2.transpose(1,2,0),0,1)) ** 2)
    # print(mse)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def inference(model_path,data_path,fpn_path,output_path):
    session = onnxruntime.InferenceSession(model_path,providers=['CUDAExecutionProvider'])
    pixel_unshuffle = PixelUnShuffle(upscale_factor=2)
    all_files = os.listdir(data_path)

    fpn = cv2.imread(fpn_path,-1)
    for i in range(len(all_files)):
        print(i)
        img_path = os.path.join(data_path,f'{str(i)}.tif')
        # data=cv2.imread(path+str(i)+'.tif',-1)
        data = cv2.imread(img_path,-1)
        if data is None:
            raise "Image does not exist, please check image path"
        data=np.clip((data-fpn)/256*64,0,255)
        
        data=torch.from_numpy(data.reshape(1,1,1024,1280)).type(torch.uint8)
        if i==0:
            pre = data
            pre_res=data
        else:
            pre_res = res

        now = time.time()
        inp1 = pixel_unshuffle(pre_res.float())
        inp2 = pixel_unshuffle(data.float())
        end = time.time()
        print(f"unshuffle time spent:{(end - now) * 1000}ms")


        # input_data = torch.cat([pre_res.float(),data.float()],1)
        input_data = torch.cat([inp1,inp2],1)
        raw_result = session.run([], {'input': input_data.cpu().numpy()})
        res = torch.tensor(raw_result[0])
        res = res.detach()
        res = torch.clamp(res,0,256).type(torch.uint8)
        img=np.array(res[0][0].cpu().detach().numpy())
        img_output_path = os.path.join(output_path,str(i)+'.bmp')
        cv2.imwrite(img_output_path,img)

if __name__ == "__main__":
    # process input data
    data_path='./noise_9'
    fpn_path='avg_2.tif'
    # process fp32
    model_path = 'edvc_downsample.onnx'
    output_path_fp32 = './output/output_downsample'
    if not os.path.exists(output_path_fp32):
        os.mkdir(output_path_fp32)

    inference(model_path,data_path,fpn_path,output_path_fp32)