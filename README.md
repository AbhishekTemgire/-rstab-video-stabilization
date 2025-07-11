# RStab Installation and Usage Guide

## 1. Miniconda Setup

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/root/miniconda3/bin:$PATH"
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Create Environment and Install Dependencies

```bash
conda create -n rstab python=3.10 -y
conda activate rstab
apt update && apt install -y git wget ffmpeg

# ⭐ CRITICAL: Install PyTorch FIRST
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

git clone https://github.com/pzzz-cv/RStab.git --recursive
cd RStab
pip install numpy==1.21.6
pip install opencv-python==4.7.0.72
pip install tensorflow==2.12.0
pip install -r requirements.txt
mkdir -p RStab_core/pretrained
mkdir -p input output
```

## 3. Verify PyTorch and CUDA

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, PyTorch: {torch.__version__}')"
```

## 4. Download Pretrained Model and Input Video

```bash
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1q3QM1damtvHLukhIOIAdv9IKm646Oj11&export=download&confirm=t" -O RStab_core/pretrained/model_255000.pth
wget --no-check-certificate "https://drive.usercontent.google.com/download?id=1zd3RJeMHGDKulGuUEBGf8zX9gkOAU8gV&export=download" -O input/original_video.mp4
```

## 5. Preprocess Input Video

```bash
ffmpeg -i input/original_video.mp4 -vf scale=1280:720 -c:v libx264 -crf 23 input/video_720p.mp4
```

## 6. Test Model Loading

```bash
python -c "import torch; torch.load('RStab_core/pretrained/model_255000.pth', map_location='cpu'); print('Model OK')"
```

## 7. Configuration

```bash
cat > RStab_core/configs/eval.txt << 'EOF'
### INPUT
rootdir = ./
ckpt_path = ./pretrained/model_255000.pth
distributed = False
## dataset
eval_dataset = nus
factor = 1
keep_size = True
height = 720
width = 1280
### TESTING
chunk_size = 300000
### RENDERING
N_importance = 0
N_samples = 3
inv_uniform = True
white_bkgd = False
neighbor_list = [-5, -3, -1, 0, 1, 3, 5]
preprocessing_model = MonST3R
sample_range_gain = 1.5
no_color_correction = False
EOF
```

## 8. Standard Workflow

```bash
cd Deep3D
python geometry_optimizer.py --video_path ../input/test.avi --output_dir ../output/Deep3D --name test
cd ../RStab_core
python rectify.py --expname test
```

## 9. MonST3R Setup

```bash
cd MonST3R
conda create -n monst3r python=3.11 cmake=3.14.0
conda activate monst3r 
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install -r requirements_optional.txt
```

## 10. Alternative Workflow

### 1. Deep3D Preprocessing

```bash
cd Deep3D
python geometry_optimizer.py --video_path ../input/video_720p.mp4 --output_dir ../output/Deep3D --name video_720p
```

### 2. MonST3R Enhancement

```bash
cd ../MonST3R
python demo.py --input ../output/Deep3D/video_720p/images --output_dir ../output/MonST3R/video_720p --seq_name output
```

### 3. RStab Stabilization

```bash
cd ../RStab_core
python rectify.py --expname video_720p
```

## 11. AWS CLI Installation

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

pip install awscli

aws configure
```

## 12. Upload Output to S3

```bash
aws s3 cp output/RStab/test/RStab_test.avi s3://your-bucket-name/rstab_stabilized.avi
```

## 13. Convert to MP4 and Upload

```bash
ffmpeg -i output/RStab/video_720p/video_720p.avi -c:v libx264 -c:a aac -crf 23 final_stabilized_720p.mp4
aws s3 cp final_stabilized_720p.mp4 s3://ditrarstab/final_stabilized_720p.mp4
```
