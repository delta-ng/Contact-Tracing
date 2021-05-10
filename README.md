# Contact-Tracing

## Installation

1. Installing torchreid : 

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
    Download all the Yolo-coco data from [here](https://drive.google.com/drive/folders/1YJymHQ9xW9w12slCPS4aq_pfqvsfAbSE?usp=sharing) and place it into the ***yolo-coco*** directory.

3. Download Model :

    We are by default using the *osnet_x1_0* model. Download the model from [here](https://drive.google.com/file/d/1tuYY1vQXReEd8N8_npUkc7npPDDmjNCV/view?usp=sharing) and place it inside the ***model*** directory. Alternatively you can also explore other models from [torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
