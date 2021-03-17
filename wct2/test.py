from wct2.transfer import WCT2
import torch
from torchvision.utils import save_image
from wct2.utils.io import open_image
import numpy as np
cpu_flag = False
device = 'cpu' if cpu_flag or not torch.cuda.is_available() else 'cuda:0'
device = torch.device(device)
option_unpool_ = 'cat5' # choices=['sum', 'cat5']
alpha_ = 1.0

_transfer_at = set()
_transfer_at.add('decoder')  # choices=['encoder', 'decoder', 'skip', 'all']

content_path = './examples/content/2019-09-26-06-07-43.jpg'
style_path = './examples/style/2020-07-28-08-12-19.jpg'
image_size = 1024
content, ori_w, ori_h = open_image(content_path, image_size, return_ori_size=True)
content = content.to(device)
style = open_image(style_path, image_size).to(device)
content_segment = np.asarray([])
style_segment = np.asarray([])
output_path = './output/resutls.png'

wct2_ = WCT2(transfer_at=_transfer_at, option_unpool=option_unpool_, device=device)
with torch.no_grad():
  img = wct2_.transfer(content, style, content_segment, style_segment, alpha=alpha_)
  img = torch.nn.functional.interpolate(img, (ori_w, ori_h), mode='bilinear', align_corners=False)
save_image(img.clamp_(0, 1), output_path, padding=0)
