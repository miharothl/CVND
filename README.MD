
conda create --name cvnd python=3.6


pip install -r env/python/requirements.txt



https://pytorch.org/get-started/previous-versions/

OSX

pip install torch==1.6.0 torchvision==0.7.0


Linux and Windows

# CUDA 10.2
pip install torch==1.6.0 torchvision==0.7.0

# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html



python -m ipykernel install --user --name cvnd --display-name "cvnd"



