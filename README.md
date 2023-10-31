# IOPRE
Interactive Object Placement with Reinforcement Learning

### Install
  First, clone the repository locally:
  ```
  git clone https://github.com/lingtianxia123/IOPRE.git
  cd IOPRE
  ```
  Then, create a virtual environment:
  ```
  conda create -n IOPRE python=3.8
  conda activate IOPRE
  ```
  Install PyTorch 1.9.1:
  ```
  conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
  ```
  Install necessary packages:
  ```
  pip install -r requirements.txt
  ```
  Install pycocotools for accuracy evaluation:
  ```
  pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  ```
  Build faster-rcnn for accuracy evaluation:
  ```
  cd faster-rcnn/lib
  python setup.py build develop
  cd ../..
  ```
### Data preparation
  Download and extract OPA dataset (https://github.com/bcmi/GracoNet-Object-Placement):
   ```
  <PATH_TO_OPA>
    background/       
    foreground/       
    composite/       
    train_set.csv     
    test_set.csv 
  ```
  Data preprocessing:
  ```
  python data/preprocess_pos_group_only.py
  python data/preprocess_pos_single.py
  ```
  or download from [baidu disk](https://pan.baidu.com/s/1ugyUzt1e3aaQCamRotXQLA) (code: y1cw)

### Inference

Download weight from [baidu disk](https://pan.baidu.com/s/1ugyUzt1e3aaQCamRotXQLA) (code: y1cw) and place it in the 'result' folder :

Generate composite image:
  ```
  python test_acc.py
  python test_lpips.py
  ```
### Evaluation
To evaluate accuracy via the classifier, please 1) download the faster-rcnn model pretrained on visual genome from [google drive](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view) (provided by [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)) to ```faster-rcnn/models/faster_rcnn_res101_vg.pth```, 2) download the pretrained binary classifier model from [bcmi cloud](https://cloud.bcmi.sjtu.edu.cn/sharing/XPEgkSHdQ) or [baidu disk](https://pan.baidu.com/s/1skFRfLyczzXUpp-6tMHArA) (code: 0qty) to ```BINARY_CLASSIFIER_PATH```, and 3) run:
  ```
  python test_acc.py
  ```
To evaluate FID score, run:
  ```
  python test_acc.py
  ```
To evaluate LPIPS score, run:
  ```
  python test_acc.py
  ```
### Citation

  If you found this code useful please cite our work as:

  ```
  @InProceedings{pmlr-v202-zhang23ag,
  title={Interactive Object Placement with Reinforcement Learning},
  author={Zhang, Shengping and Meng, Quanling and Liu, Qinglin and Nie, Liqiang and Zhong, Bineng and Fan, Xiaopeng and Ji, Rongrong},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  pages={41510--41522},
  year={2023},
  volume={202}
}
    ```




