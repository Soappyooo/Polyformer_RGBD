# PolyFormerRGBD
[PolyFormer](https://github.com/amazon-science/polygon-transformer) with RGBD input. 

## Installation
```bash
# create conda environment
conda create -n polyformer_rgbd python=3.7.4 -y
conda activate polyformer_rgbd
# install dependencies
conda install -c conda-forge ld_impl_linux-64 -y
pip install pip==21.2.4
pip install -r requirements.txt
```

## Download Checkpoint
The RGBD checkpoint can be downloaded from [OneDrive](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/fandongxuan24_mails_ucas_ac_cn/EWjgenjkHyVPmCXY0HW_tS8B6VgA8f_mIoJyMxfYlEenbA?e=PK49Ive) (2.4GB).  
Place the checkpoint in `./checkpoints/` folder.  

## Run the Inference Demo
```bash
python res.py
```

## Acknowledgement
This repository is based on the [PolyFormer](https://github.com/amazon-science/polygon-transformer). We thank the authors for their great work.