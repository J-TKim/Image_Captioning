# Image_Captioning
Image_Captioning Project


## Step
### 1. Clone COCO API repository
```shell
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
```

### 2. Clone Image_Captioning dataset
```shell
$ git clone https://github.com/J-TKim/Image_Captioning.git
cd Image_Captioning
```

### 3. Download the dataset
```shell
pip install -r requirements.txt
chmod +x download.sh
./download.sh
```

### 4. Prerocessing
```shell
python build_vocab.py
python resize.py
```

### 5. Train the model
```shell
python train.py
```

### 6. Test the model
```shell
python sample.py --image='png/example.png'
```
-----
## If you are training with korean data set, please refer to the method below.

### 1. download dataset from AIHUB
[Aihub captioning data](https://aihub.or.kr/keti_data_board/visual_intelligence)

### 2. move ko caption dataset to ./data/annotations

### 3. Preprocessing
```shell
python build_vocab.py --ko data/annotations/en2ko.json
python resize.py
```

### 4. Train the model
```shell
python train.py --ko data/annotations/en2ko.json
```

### 5. Test the model
```shell
python sample.py --image='png/example.png'
```