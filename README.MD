
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


# Project 2
http://cocodataset.org/

https://github.com/cocodataset/cocoapi.git

conda activate cvnd

pip install cython

cd cocoapi/PythonAPI

make

## get data

### project

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

http://images.cocodataset.org/zips/train2014.zip
http://images.cocodataset.org/zips/val2014.zip
http://images.cocodataset.org/zips/test2014.zip

### cocoapi demo

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip


# check cocoapi notebooks

pip install scikit-image


pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
