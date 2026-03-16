# [NTIRE 2026 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2026_ESR/blob/main/figs/logo.png" width="400px"/> 
</div>

## Quick test
How to test the model?
1. Create a new environment
   
```bash
conda create -n SPAN-Mamba python=3.10.13
conda activate SPAN-Mamba
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3.post1/causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.1.3.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 
```


2. Clone the repo
```bash
git clone https://github.com/Zty031220/NTIRE2026-EfficientSR-zty77.git
```

3. Install dependent packages
```bash
cd NTIRE2026-EfficientSR-zty77
pip install -r requirements.txt
```
4. Testing Command
```bash
chmod +x run.sh
./run.sh
```
