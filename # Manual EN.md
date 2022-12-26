# Manual for real time semantic segmentation


##  **1. Specification**
### **Hardware**
* CPU: Intel Xeon W-2245 CPU 3.90GHz x 16
* GPU: NVDIA GeForce RTX 2080ti
* RAM 32GB

### **Software**
* OS: Ubuntu 18.04.6 LTS
* CUDA: 10.2
* CUDNN: 8.0.3
* Torch: 1.6.0

## **2. Train**

### **2.1. Dataset**  
Join in [Cityscapes](https://www.cityscapes-dataset.com/) and download gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip

### **2.2. Install CUDA/CUDNN**
(1) CUDA Download
```python
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```
(2) Install CUDA Toolkit, Samples, Documentation except Driver

(3) CUDA Absolute Path
```python
gedit ~/.bashrc
export CUDA_HOME=/usr/local/cuda-10.2
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

(4) Confirm Installation
```python
nvcc -V
```

(5) [CUDNN Access](https://developer.nvidia.com/rdp/cudnn-archive)
Download cuDNN v8.0.3 (Aug 26, 2020), for CUDA 10.2 cuDNN Library for Linux
Unzip tar files

(6) CUDNN Install
```python
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### **2.3. RegSeg**

#### 2.3.0 Virtual env & Torch install
Since lower version of torchvision does not support InterpolationMode.NEAREST, we installed torch 1.6.0 with torchvision 0.10.0
```python
conda create -n RegSeg python=3.7
pip install torchvision==0.10.0
pip install torch==1.6.0
```

#### 2.3.1 Clone repo
```python
git clone https://github.com/RolandGao/RegSeg
cd RegSeg
```
#### 2.3.2 Install dependencies
```python
pip install -r requirements.txt
```

#### 2.3.3 Dataset Preprocess
Create the folders named cityscapes_dataset, then unzip gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip in the folders
```python
CITYSCAPES_DATASET=cityscapes_dataset csCreateTrainIdLabelImgs
```

#### 2.3.4 Split Revision
To implement with torch 1.6.0, tensor_split function of block.py line 76 should be replaced with below.
```python
size = x.size()
split = size[1] // 2
x=torch.split(x, split, dim=1)
```


#### 2.3.5 Modify config file (Unify data augmentation)
Modify /config/cityscapes_1000epochs.yaml as below.
train_crop_size: ~~[768, 768]~~ -> [768, 1536]
aug_mode: ~~randaug_reduced~~ -> baseline  

#### 2.3.6 Training
Annotate [line 406, 407](https://github.com/RolandGao/RegSeg/blob/40f8c0bab7048eb5bb8bba6c76393275e61f7050/train.py#L406-L407) of train.py
```python
python train.py
```

#### 2.3.7 Validate
Unannotate line 407 of train.py then annotate line 408
```python
python train.py
```

#### 2.3.7 Onxx export
Before export, you should revise the path of pretrained pth file
```python
import random
import torch
from model import RegSeg

model=RegSeg(
name="exp48_decoder26",
num_classes=19,
pretrained="your checkpoint path"
)
model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 1024, 2048, device='cuda')
torch.onnx.export(model, dummy_input, "regseg.onnx", verbose=False, export_params=True, opset_version=11)
```


### **2.4. DDRNet**

#### 2.4.0 Virtual env
Use virtual env of RegSeg as it is.

#### 2.4.1 Modify config file (Unify data augmentation)
Modify /config/cityscapes_competitor_1000epochs.yaml as below.

train_crop_size: ~~[768, 768]~~ -> [768, 1536]
aug_mode: ~~randaug_reduced~~ -> baseline  

#### 2.4.2 Training
Annotate [line 406, 407](https://github.com/RolandGao/RegSeg/blob/40f8c0bab7048eb5bb8bba6c76393275e61f7050/train.py#L406-L407) of train.py
```python
python train.py
```

#### 2.3.3 Validate
Unannotate line 407 of train.py then annotate line 408
```python
python train.py
```

#### 2.3.4 Onxx export
Before export, you should revise the path of pretrained pth file
```python
import random
import torch
from model import RegSeg

model=RegSeg(
name="exp48_decoder26",
num_classes=19,
pretrained="your checkpoint path"
)
model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 1024, 2048, device='cuda')
torch.onnx.export(model, dummy_input, "regseg.onnx", verbose=False, export_params=True, opset_version=11)
```



### **2.4. STDC-Seg**

#### 2.4.0 Virtual env
```python
conda create -n stdc python=3.7
```

#### 2.4.1 Clone repo
```python
git clone https://github.com/MichaelFan01/STDC-Seg.git
cd STDC-Seg
```
#### 2.4.2 Install dependencies
```python
pip install -r requirements.txt
pip install ninja
```

#### 2.4.3 Dataset relocate  
Create folder named data in STDC-Seg directory  
Unzip gtFine, lefIg8bit in data folder 
```python
ln -s /path_to_data/cityscapes/gtFine data/gtFine
ln -s /path_to_data/leftImg8bit data/leftImg8bit
```

#### 2.4.4 Pretrained weights
Locate [STDCNet813M_73.91.tar, STDCNet1446_76.47.tar](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing) in STDC-Seg/checkpoints

#### 2.4.5 Modify scripts (Unify data augmentation)
STDC-Seg/cityscapes.py  
Modify [line 20, 21](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/cityscapes.py#L20-L21)
 ```python
def __init__(self, rootpth, cropsize=(1536, 768), mode='train', 
    randomscale = random.uniform(0.390625, 1.5625), *args, **kwargs):
```

Modify [line 72~84](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/cityscapes.py#L68-L84)
```python
self.trans_train = Compose([
    HorizontalFlip(),
    RandomScale(randomscale),
    RandomCrop(cropsize)
    ])
```

STDC-Seg/transform.py  
Modify [line 60](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/transform.py#L60)
```python
scale = self.scales
```

Modify [line 116~118](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/transform.py#L116-L118)
```python
flip = HorizontalFlip(p = 0.5)
crop = RandomCrop((1536, 768))
rscales = RandomScale(random.uniform(0.390625, 1.5625))
```

STDC-Seg/evaluation.py  
Annotate [line 277, 278](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/evaluation.py#L277-L278)
In case of train STDC1, replace the block with below
```python
#STDC1-Seg100
evaluatev0('/home/username/STDC-Seg/checkpoints/train_STDC1-Seg/pths/model_maxmIOU100.pth', dspth='./data', backbone='STDCNet813', scale=1, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
```
In case of train STDC2, replace the block with below
```python
#STDC2-Seg100
evaluatev0('/home/username/STDC-Seg/checkpoints/train_STDC2-Seg/pths/model_maxmIOU100.pth', dspth='./data', backbone='STDCNet1446', scale=1, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
```

STDC-Seg/train.py  
Add Random library after line 22
```python
import random
```

Modify [line 175, 176](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L175-L176)
```python
cropsize = [1536, 768]
randomscale = random.uniform(0.390625, 1.5625)
```

Modify [line230, 231](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L230)
```python
maxmIOU100 = 0.
```

 Modify [line 377~412](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L377-L412)
```python
logger.info('compute the mIOU')
with torch.no_grad():
    single_scale= MscEvalV0(scale=1)
    mIOU100 = single_scale(net, dlval, n_classes)

save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU100_{}.pth'
.format(it+1, str(round(mIOU100,4))))

state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
if dist.get_rank()==0: 
    torch.save(state, save_pth)

logger.info('training iteration {}, model saved to: {}'.format(it+1, save_pth))

if mIOU100 > maxmIOU100:
    maxmIOU100 = mIOU100
    save_pth = osp.join(save_pth_path, 'model_maxmIOU100.pth'.format(it+1))
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank()==0: 
        torch.save(state, save_pth)
        
    logger.info('max mIOU model saved to: {}'.format(save_pth))

logger.info('mIOU100 is: {}'.format(mIOU100))
logger.info('maxmIOU100 is: {}.'.format(maxmIOU100))
```

If *RuntimeError: CUDA out of memory.* occured, 
Decrease n_img_per_gpu of [line 59](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L59) suitable for your GPU capacity (we set 2 with 2080ti)

#### 2.4.6 Excute Training  
For train STDC1
```python
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
--nproc_per_node=3 train.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar
```
For train STDC2
```python
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch \
--nproc_per_node=3 train.py \
--respath checkpoints/train_STDC2-Seg/ \
--backbone STDCNet1446 \
--mode train \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet1446_76.47.tar
```

#### 2.4.7 Validate
```
CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

#### 2.4.8 Onnx Export
In case of STDC2, backbone of evaluatev0 should be STDCnet1446
```python
from models.model_stages import BiSeNet
import torch
import torch.nn as nn

def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=1.0, use_boundary_2=False,
               use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    n_classes = 19
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
                  use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
                  use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
                  use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()

    dummy_input = torch.randn(1, 3, 1024, 2048, device='cuda')
    torch.onnx.export(net, dummy_input, "test.onnx", opset_version=11)

    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        single_scale(net, dl, 19)


if __name__ == "__main__":
    evaluatev0('PTH 파일 경로', dspth='데이터셋 경로', backbone='STDCNet813', scale=1.0, use_boundary_2=False,
               use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)

```


### **2.5. FasterSeg**

#### 2.5.1 Prerequisites
In case of FasterSeg, we were unable to train in same environment above.  
Therefore we used CUDA 10.0, CUDNN 7.3.0, TensoRT-5.0 for train.  
The way to install multiple CUDA version is as [such](https://m31phy.tistory.com/125).

#### 2.5.2 Clone repo
```python
git clone https://github.com/chenwydj/FasterSeg.git
cd FasterSeg
```

#### 2.5.3 Install dependencies
```python
pip install -r requirements.txt
python -m pip install cityscapesscripts  
```

#### 2.5.4 Install PyCUDA
Refer to 3.1 PyCUDA

#### 2.5.5 Install TensorRT
Refer to 3.2 TensorRT

#### 2.5.6 Data Preprocess
Create folder named data in FasterSeg directory.  
Move the text files in [FasterSeg/tools/datasets/cityscapes/ ](https://github.com/VITA-Group/FasterSeg/tree/master/tools/datasets/cityscapes) to data folder.
Unzip gtFine, lefImg8bit of cityscapes dataset in data folder.  
Download [Data prerpoces file](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) in data folder and implement it.  
```python
python createTrainIdLabelImgs.py  
```

#### 2.5.7 Search
Modity [line 25](https://github.com/VITA-Group/FasterSeg/blob/478b0265eb9ab626cfbe503ad16d2452878b38cc/search/config_search.py) of FasterSeg/search/config_search.py as below and implement it.
```python
cd search
C.dataset_path = "/home/username/FasterSeg/data/"
C.train_scale_array = [0.390625, 1, 1.5625]

CUDA_VISIBLE_DEVICES=0 python train_search.py
```
if RuntimeError: CUDA out of memory occured, decrease batch size at config_search.py

#### 2.5.8 Teacher Train
Copy searched folder that contains arch_0.pt and arch_1.pth to train directory.  
Modify config_train.py as below.
```python
cd FasterSeg/train
C.dataset_path = "/home/usename/FasterSeg/data/"
C.mode = "teacher"
C.train_scale_array = [0.390625, 1, 1.5625]
C.load_path = "/home/usename/FasterSeg/train/search-pretrain-(Set the name of your searcher folder here)"
C.image_height = 768
C.image_width = 1536

CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.9 Student Train
Modify config_train.py as below.  
```python
C.mode = "student"
C.train_scale_array = [0.390625, 1, 1.5625]
C.load_path = "/home/username/FasterSeg/train/search-pretrain-(Set the name of your searcher folder here)"
C.teacher_path = "/home/username/FasterSeg/train/train-768x1536_teacher_(Set the name of your teacher folder here)"
C.image_height = 768
C.image_width = 1536

CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.10 Validate
Modify config_train.py as below  
```python
#Line 107
C.is_eval = True

CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.11 Onnx Export
```python
cd FasterSeg/latency
CUDA_VISIBLE_DEVICES=0 python run_latency.py
```

#### 2.5.12 Issue Report
 If ValueError: Object arrays cannot be loaded when allow_pickle=False occured,  
 Please modify train/seg_oprs.py, train/operation.py, latency/seg_oprs.py, latency/operation.py, search/seg_oprs.py, search/operation.py as below.
 ```python
 #Before
 latency_lookup_table = np.load(table_file_name).item()
 #After
 latency_lookup_table = np.load(table_file_name, allow_pickle=True).item()
 ```

## **3. Validation**

### **3.1. Pycuda**
Install in Terminal.
```python
pip install pycuda
```

### **3.2. TensorRT**
#### 3.2.1 Installation
Download TensorRT 7.1.3.4 for Ubuntu 18.04 and CUDA 10.2 TAR package from https://developer.nvidia.com/nvidia-tensorrt-7x-download  
```python
tar xzvf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
export LD_LIBRARY_PATH=/home/im/TensorRT-7.1.3.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
cd TensorRT-7.1.3.4/python
sudo pip install tensorrt-7.1.3.4-cp37-none-linux_x86_64.whl
cd ../uff
sudo pip install uff-0.6.9-py2.py3-none-any.whl
cd ../graphsurgeon
sudo pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
#Check installation
python
import tensorrt
```
#### 3.2.2 Trtexec
Onnx file can be tranlated in egnine file with TensorRT.  
```python
/home/username/TensorRT-7.1.3.4/bin
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine
#FP16
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine --fp16
#INT8
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine --int8
```

### **3.3. Xaiver NX**
#### 3.3.1. Setup
Download SD Card Image(4.5.1 in our case) from https://developer.nvidia.com/embedded/downloads#?search=SD%20Card%20Image  
Make image file in microSD card as [instruction](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#prepare)    
Insert microSD card in your board and boot.  

#### 3.3.2 Package update & Install virtual env
```python
sudo apt update && sudo apt upgrade
virtualenv venv
source venv/bin/activate
```
#### 3.3.3 PyTorch, torchvision
```python
#torch install
wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl
#torchvision install
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch <version> https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.7.0
python3 setup.py install --user
cd ../  # attempting to load torchvision from build dir will result in import error
```

```python
#install verification
python3
import torch
print(torch.__version__)
print('CUDA available: ' + str(torch.cuda.is_available()))
print('cuDNN version: ' + str(torch.backends.cudnn.version()))
import torchvision
print(torchvision.__version__)
```

#### 3.3.4 Open CV
```
sudo apt install build-essential cmake pkg-config -y 

sudo apt install libjpeg-dev libtiff5-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libx265-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran python3-dev

sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-pulseaudio libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.5.zip

unzip opencv.zip
cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=OFF \
-D WITH_IPP=OFF \
-D WITH_1394=OFF \
-D BUILD_WITH_DEBUG_INFO=OFF \
-D BUILD_DOCS=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D WITH_QT=OFF \
-D WITH_GTK=ON \
-D WITH_OPENGL=OFF \
-D WITH_V4L=ON  \
-D WITH_FFMPEG=ON \
-D WITH_XINE=ON \
-D WITH_GSTREAMER=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \
-D PYTHON3_NUMPY_INCLUDE_DIR=/usr/local/lib/python3.6/dist-packages/numpy/core/include \
-D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.6/dist-packages \
-D PYTHON3_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.6m.so \
../

make -j 6
sudo -H make install
```

#### 3.3.5 Speed test with memory buffer
```
cd /usr/src/tensorrt/bin
./trtexec --loadEngine=your/engine/path
```
