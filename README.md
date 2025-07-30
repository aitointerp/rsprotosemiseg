# Environment
- Python 3.8
- Setup python environment：

```
pip install -r requirements.txt
```
# Datasets
- Please download three original datasets：LoveDA,DGLCC,Postdam.
- For the DGLCC and Postdam datasets, the original images are cropped into fixed-size patches of 1024×1024 pixels using a sliding window.
# Pretrained
The uses ResNet-101 pretrained on ImageNet, please download from [here](https://download.pytorch.org/models/resnet101-63fe2227.pth)  Remember to change the directory in corresponding python file.
```
# Acknowledgement
- PRCL: https://github.com/Haoyu-Xie/PRCL Thanks a lot for their splendid work!
