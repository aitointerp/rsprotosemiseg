import shutup
shutup.please()
import torch
import torchvision.models as models
from torch.nn import functional as F

import numpy as np
from PIL import Image
from generalframeworks.networks.deeplabv3.deeplabv3_gmm import DeepLabv3Plus_with_un
from generalframeworks.augmentation.transform import transform_test
import torchvision.transforms.functional as TF

# ++++++++++++++++++++ Utils +++++++++++++++++++++++++


def create_potsdam_label_colormap():
  """Creates a label colormap used in Pascal segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = 255 * np.ones((256, 3), dtype=np.uint8)
  colormap[0] = [255, 255, 255]
  colormap[1] = [255, 0, 0]
  colormap[2] = [255, 255, 0]
  colormap[3] = [0, 255, 0]
  colormap[4] = [0, 255, 255]
  colormap[5] = [0, 0, 255]
  return colormap

def color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)


# ++++++++++++++++++++ Pascal VOC Visualisation +++++++++++++++++++++++++
# Initialization
im_size = [512, 512]
root = 'D:/Python/RSProtoSemiSeg/dataset/Potsdam'
num_segments = 6
device = torch.device("cuda")
model = DeepLabv3Plus_with_un(models.resnet101(), num_classes=num_segments).to(device)

# Load checkpoint
checkpoint = torch.load('/.pth', map_location='cuda')
model.load_state_dict(checkpoint['model'])

# Switch to eval mode
model.eval()

# Generate color map for visualisation
colormap = create_potsdam_label_colormap()

# Visualise image in validation set

# Load images and pre-process
with open(root + '/test_filename.txt') as f:
    idx_list = f.read().splitlines()
for id in idx_list:
    print('Image {} start!'.format(id))
    im = Image.open(root + '/test/image/{}.png'.format(id))
    #im.save(root+'/vis_result/image/{}.png'.format(id))
    gt_label = Image.open(root+'/test/label/{}.png'.format(id))
    image,label,im_tensor, label_tensor = transform_test(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    #image.save(root+'/vis_result/5%/image/{}.png'.format(id))
    #label.save(root+'/vis_result/5%/label/{}.png'.format(id))
    im_w, im_h = image.size


    # Move input data to CUDA device
    im_tensor = im_tensor.to(device)

    # Inference
    logits, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_prcl = torch.max(torch.softmax(logits, dim=1), dim=1)

    #根据类别标签映射为灰度图像
    gray_img = label_prcl.byte()  # 将标签张量转换为字节类型张量

    # 创建与原始图像相同大小的灰度图像
    gray_img_pil = Image.fromarray(gray_img.squeeze().cpu().numpy(), mode='L')

    # 保存生成的灰度图像
    gray_img_pil.save(root + '/vis_result/5%/prcl_gmm_gray/{}.png'.format(id))
