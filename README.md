# Contact-Tracing

## Installation

1. Installing Torch-Reid : 

  ```
  git clone https://github.com/KaiyangZhou/deep-person-reid.git

  # create environment
  cd deep-person-reid/
  conda create --name torchreid python=3.7
  conda activate torchreid

  # install dependencies
  # make sure `which python` and `which pip` point to the correct path
  pip install -r requirements.txt

  # install torch and torchvision (select the proper cuda version to suit your machine)
  conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

  # install torchreid (don't need to re-build it if you modify the source code)
  python setup.py develop
  ```

2. Download Yolo File :
  Create a repo ***yolo-coco*** and download all the Yolo-coco data from [here](https://drive.google.com/drive/folders/1YJymHQ9xW9w12slCPS4aq_pfqvsfAbSE?usp=sharing) and place it into the yolo-coco directory.
