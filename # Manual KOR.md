# Manual for real time semantic segmentation


##  **1. 사양**
### **하드웨어**
* 프로세서: Intel Xeon W-2245 CPU 3.90GHz x 16
* 그래픽: NVDIA GeForce RTX 2080ti
* 메모리: RAM 32GB

### **소프트웨어**
* OS: Ubuntu 18.04.6 LTS
* CUDA: 10.2
* CUDNN: 8.0.3
* Torch: 1.6.0

## **2. 학습**

### **2.1. 데이터 준비**  
[Cityscapes](https://www.cityscapes-dataset.com/) 홈페이지에서 회원 가입 후 gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip를 다운로드

### **2.2. CUDA/CUDNN 설치**
(1) CUDA 다운로드 후 실행  
```python
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```
(2) Driver를 제외한 CUDA Toolkit, Samples, Documentation만 설치

(3) CUDA 환경 및 경로 설정
```python
gedit ~/.bashrc
export CUDA_HOME=/usr/local/cuda-10.2
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

(4) 재부팅 후 설치 확인
```python
nvcc -V
```

(5) [CUDNN 접속](https://developer.nvidia.com/rdp/cudnn-archive) 및 다운로드  
Download cuDNN v8.0.3 (Aug 26, 2020), for CUDA 10.2
cuDNN Library for Linux 다운로드  
다운로드 된 tar 파일 압축 해제  

(6) CUDNN 설치
```python
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### **2.3. RegSeg**

#### 2.3.0 Virtual env & Torch install
/torchvision의 InterpolationMode.NEAREST는 하위 호환이 지원되지 않아 다음과 같이 설치합니다.
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
/RegSeg 위치에 cityscapes_dataset 이름의 폴더 생성 후 그 안에 gtFine_trainvaltest.zip, leftImg8bit_trainvaltest.zip 압축 해제
```python
CITYSCAPES_DATASET=cityscapes_dataset csCreateTrainIdLabelImgs
```

#### 2.3.4 Split Revision
torch 1.6 버전에서 실행하기 위해 tensor_split 함수를 수정해야 합니다.  
block.py의 76번 줄을 다음과 같이 수정합니다.
```python
size = x.size()
split = size[1] // 2
x=torch.split(x, split, dim=1)
```


#### 2.3.5 Modify config file (Unify data augmentation)
/config/cityscapes_1000epochs.yaml 파일을 다음과 같이 수정

train_crop_size: ~~[768, 768]~~ -> [768, 1536]
aug_mode: ~~randaug_reduced~~ -> baseline  

#### 2.3.6 Training
train.py의 [406, 407번 줄](https://github.com/RolandGao/RegSeg/blob/40f8c0bab7048eb5bb8bba6c76393275e61f7050/train.py#L406-L407)을 주석 처리
```python
python train.py
```

#### 2.3.7 Validate
train.py의 407번 줄 주석 해제 후 408번 줄을 주석 처리
```python
python train.py
```

#### 2.3.7 Onxx export
변환 전 pretrained pth 파일의 경로를 입력해준다.
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
RegSeg의 가상환경에서 그대로 실험을 진행합니다.

#### 2.4.1 Modify config file (Unify data augmentation)
/config/cityscapes_competitor_1000epochs.yaml 파일을 다음과 같이 수정

train_crop_size: ~~[768, 768]~~ -> [768, 1536]
aug_mode: ~~randaug_reduced~~ -> baseline  

#### 2.4.2 Training
train.py의 [406, 407번 줄](https://github.com/RolandGao/RegSeg/blob/40f8c0bab7048eb5bb8bba6c76393275e61f7050/train.py#L406-L407)을 주석 처리
```python
python train.py
```

#### 2.3.3 Validate
train.py의 407번 줄 주석 해제 후 408번 줄을 주석 처리
```python
python train.py
```

#### 2.3.4 Onxx export
변환 전 pretrained pth 파일의 경로를 입력해준다.
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
STDC-Seg 폴더 내에 data 폴더 생성  
STDC-Seg/data 폴더 내에 cityscapes 데이터셋의 gtFine, lefIg8bit 압축 해제  
```python
ln -s /path_to_data/cityscapes/gtFine data/gtFine
ln -s /path_to_data/leftImg8bit data/leftImg8bit
```

#### 2.4.4 Pretrained weights
[STDCNet813M_73.91.tar, STDCNet1446_76.47.tar](https://drive.google.com/drive/folders/1wROFwRt8qWHD4jSo8Zu1gp1d6oYJ3ns1?usp=sharing) 파일을 STDC-Seg/checkpoints에 준비

#### 2.4.5 Modify scripts (Unify data augmentation)
*train.py에서 cityscapes.py를 호출, 그리고 cityscapes.py에서 transform.py를 호출합니다.*  
STDC-Seg/cityscapes.py  
[20, 21번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/cityscapes.py#L20-L21)을 다음과 같이 수정
 ```python
def __init__(self, rootpth, cropsize=(1536, 768), mode='train', 
    randomscale = random.uniform(0.390625, 1.5625), *args, **kwargs):
```

[72~84번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/cityscapes.py#L68-L84)을 다음과 같이 수정
```python
self.trans_train = Compose([
    HorizontalFlip(),
    RandomScale(randomscale),
    RandomCrop(cropsize)
    ])
```

STDC-Seg/transform.py  
[60번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/transform.py#L60)을 다음과 같이 수정
```python
scale = self.scales
```

[116~118번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/transform.py#L116-L118)을 다음과 같이 수정
```python
flip = HorizontalFlip(p = 0.5)
crop = RandomCrop((1536, 768))
rscales = RandomScale(random.uniform(0.390625, 1.5625))
```

STDC-Seg/evaluation.py  
[277, 278번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/evaluation.py#L277-L278)을 주석 처리 후  
STDC1 학습할 때는 제일 밑에 아래 블록으로 대체 **사용자 이름 주의**
```python
#STDC1-Seg100
evaluatev0('/home/사용자 이름/STDC-Seg/checkpoints/train_STDC1-Seg/pths/model_maxmIOU100.pth', dspth='./data', backbone='STDCNet813', scale=1, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
```
STDC2 학습할 때는 제일 밑에 아래 블록으로 대체 **사용자 이름 주의**
```python
#STDC2-Seg100
evaluatev0('/home/im/사용자 이름/checkpoints/train_STDC2-Seg/pths/model_maxmIOU100.pth', dspth='./data', backbone='STDCNet1446', scale=1, use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_boundary_16=False)
```

STDC-Seg/train.py  
Random 라이브러리 추가 (22번 줄 뒤로)
```python
import random
```

[175, 176번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L175-L176)을 다음과 같이 수정
```python
cropsize = [1536, 768]
randomscale = random.uniform(0.390625, 1.5625)
```

[230, 231번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L230)을 다음과 같이 수정
```python
maxmIOU100 = 0.
```

 [377~412번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L377-L412)을 다음과 같이 수정  
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

*RuntimeError: CUDA out of memory.* 가 발생할 경우
[59번 줄](https://github.com/MichaelFan01/STDC-Seg/blob/59ff37fbd693b99972c76fcefe97caa14aeb619f/train.py#L59)의 n_img_per_gpu에 해당하는 16을 각자의 GPU에 알맞게 줄여줍니다. (2080ti 기준 2로 줄였습니다.)

#### 2.4.6 Excute Training  
STDC1을 학습할 때
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
STDC2을 학습할 때
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
STDC2의 경우 backbone을 STDCnet1446으로 수정해야 합니다.
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
본 논문의 github repository와 논문 상의 실험 환경이 상이하여 두 환경에서 진행한 결과,  
학습은 논문에 기술된 환경에서 진행해야 하는 것을 확인하였습니다.  
따라서 본 논문의 학습은 CUDA 10.0, CUDNN 7.3.0, TensoRT-5.0 버전을 사용하였습니다.  
한 컴퓨터에 여러 CUDA 버전을 설치하는 방법은 [다음](https://m31phy.tistory.com/125)과 같습니다.

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
3.1 PyCUDA 항목 참조

#### 2.5.5 Install TensorRT
3.2 TensorRT 항목 참조

#### 2.5.6 Data Preprocess
FasterSeg 디렉토리 상에 data 폴더 생성  
[FasterSeg/tools/datasets/cityscapes/ ](https://github.com/VITA-Group/FasterSeg/tree/master/tools/datasets/cityscapes)상에 있는 텍스트 파일 3개를 data 폴더 내로 이동  
data 폴더 내에 cityscapes 데이터셋의 gtFine, lefImg8bit 압축 해제  
data 폴더 내에 [데이터 전처리 파일](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) 다운로드 및 실행  
```python
python createTrainIdLabelImgs.py  
```
#### 2.5.7 Search
FasterSeg/search/config_search.py에서 [25번 줄](https://github.com/VITA-Group/FasterSeg/blob/478b0265eb9ab626cfbe503ad16d2452878b38cc/search/config_search.py)을 다음과 같이 수정 후 실행  
```python
cd search
# 25번 줄 데이터 경로 수정 (사용자 이름은 PC에 등록된 이름)
C.dataset_path = "/home/사용자 이름/FasterSeg/data/"
# 60번 줄 랜덤 스케일 비율 수정
C.train_scale_array = [0.390625, 1, 1.5625]

# 실행
CUDA_VISIBLE_DEVICES=0 python train_search.py
```
만일 RuntimeError: CUDA out of memory 발생할 경우 config_search.py에서 배치 크기를 줄여서 해결

#### 2.5.8 Teacher Train
arch_0.pt와 arch__1.pth가 포함되어 있는 searched folder를 train 디렉토리로 복사
config_train.py을 다음과 같이 수정  
```python
cd FasterSeg/train
#26번 줄 데이터 경로 수정
C.dataset_path = "/home/사용자 이름/FasterSeg/data/"
#77번 줄 학습 모드 변경
C.mode = "teacher"
#63번 줄 랜덤 스케일 비율 수정
C.train_scale_array = [0.390625, 1, 1.5625]
#84번 줄 search 결과 경로 수정
C.load_path = "/home/사용자 이름/FasterSeg/train/search-pretrain-(Set the name of your searcher folder here)"
#88, 89번 줄 crop size 수정
C.image_height = 768
C.image_width = 1536
#실행
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.9 Student Train
config_train.py을 다음과 같이 수정  
```python
#77번 줄 학습 모드 변경
C.mode = "student"
#63번 줄 랜덤 스케일 비율 수정
C.train_scale_array = [0.390625, 1, 1.5625]
#96번 줄 search 결과 경로 수정
C.load_path = "/home/사용자 이름/FasterSeg/train/search-pretrain-(Set the name of your searcher folder here)"
#97번 줄 teacher 결과 경로 수정
C.teacher_path = "/home/사용자 이름/FasterSeg/train/train-768x1536_teacher_(Set the name of your teacher folder here)"
#101, 102번 줄 crop size 수정
C.image_height = 768
C.image_width = 1536
#실행
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.10 Validate
config_train.py을 다음과 같이 수정 
```python
#107번줄
C.is_eval = True
#실행
CUDA_VISIBLE_DEVICES=0 python train.py
```

#### 2.5.11 Onnx Export
onnx 파일 변환 후 메모리 버퍼를 사용하여 FPS를 측정할 수 있습니다.
```python
cd FasterSeg/latency
CUDA_VISIBLE_DEVICES=0 python run_latency.py
```

#### 2.5.12 Issue Report
 ValueError: Object arrays cannot be loaded when allow_pickle=False 에러 메시지가 발생할 경우 다음과 같이 코드를 수정하면 됩니다.  
 train/seg_oprs.py, train/operation.py, latency/seg_oprs.py, latency/operation.py, search/seg_oprs./y, search/operation.py에서 해당 부분 수정
 ```python
 #수정 전
 latency_lookup_table = np.load(table_file_name).item()
 #수정 후
 latency_lookup_table = np.load(table_file_name, allow_pickle=True).item()
 ```

## **3. 검증**

### **3.1. Pycuda**
터미널에서 바로 설치할 수 있다.
```python
pip install pycuda
```

### **3.2. TensorRT**
#### 3.2.1 Installation
https://developer.nvidia.com/nvidia-tensorrt-7x-download  
위 홈페이지에서 NVIDIA 로그인 후 TensorRT 7.1.3.4 for Ubuntu 18.04 and CUDA 10.2 TAR package를 다운받는다.  
```python
#다운로드된 디렉토링에서 압축 해제
tar xzvf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
#이후 다음과 같이 절대 경로를 지정해준다.
export LD_LIBRARY_PATH=/home/im/TensorRT-7.1.3.4/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#whell file 설치
cd TensorRT-7.1.3.4/python
sudo pip install tensorrt-7.1.3.4-cp37-none-linux_x86_64.whl
cd ../uff
sudo pip install uff-0.6.9-py2.py3-none-any.whl
cd ../graphsurgeon
sudo pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
#설치 확인
python
import tensorrt
```
#### 3.2.2 Trtexec
TensorRT를 통해 onnx 파일을 engine 파일로 변환할 수 있습니다.  
```python
/home/사용자 이름/TensorRT-7.1.3.4/bin
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine
#FP16 변환
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine --fp16
#INT8 변환
./trtexec --onnx=onnx_path/file.onnx --saveEngine=save_path/file.engine --int8
```

### **3.3. Xaiver NX**
#### 3.3.1. 시작하기
https://developer.nvidia.com/embedded/downloads#?search=SD%20Card%20Image  
위 홈페이지에서 NVIDIA 로그인 후 자신에게 알맞는 버전의 SD Card Image를 다운받는다. (4.5.1 기준 작성)  
https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit#prepare  
사용하는 운영체제에 해당하는 안내에 따라 microSD card에 이미지를 작성한다.  
microSD card를 보드에 삽입하여 부팅한다.  
#### 3.3.2 패키지 업데이트 및 가상 환경 설치
```python
sudo apt update && sudo apt upgrade
virtualenv venv
source venv/bin/activate
```
#### 3.3.3 PyTorch, torchvision 설치
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

#### 3.3.4 Open CV 설치
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

#### 3.3.5 속도 측정
```
cd /usr/src/tensorrt/bin
./trtexec --loadEngine=your/engine/path
```
