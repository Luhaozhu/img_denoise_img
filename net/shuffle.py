import torch
import torch.nn as nn

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
class model_L(torch.nn.Module):
    def __init__(self):
        super(model_L, self).__init__()
        self.down=PixelUnShuffle(upscale_factor=2)
        self.up = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, inp):
        # x=self.down(inp)
        out=self.up(inp)
        return out
    

if __name__ == "__main__":
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    from torch.autograd import Variable

    device = torch.device('cpu')
    model = model_L()

    onnx_path='shuffle.onnx'
    tensor = torch.randn(1, 4, 512, 640)
    # dummy_input = [torch.randn(1, 2, 1024, 1280)
    torch_out = torch.onnx.export(model,tensor,onnx_path,input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})
    # 分析FLOPs
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total()/1e9)

    # 分析parameters
    print(parameter_count_table(model))