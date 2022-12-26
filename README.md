
# Bird Fine-Grained Classification

# File Directory
```bash
├── main.py
├── Net.py
├── train.py
├── train_utils.py
├── class_name_embedding.py
├── ClassnameProcessing.py
├── test.py
├── CUB_200_2011
│   ├── images
│   │   ├── 001.Black_footed_Albatross
│   │   │   ├── ***
│   │   │   └── Black_Footed_Albatross_0001_796111.jpg
│   │   ├── ***
│   │   └── 200.Common_Yellowthroat
│   │   │   ├── ***
│   │   │   └── Common_Yellowthroat_0126_190407.jpg
│   ├── attributes
│   ├── classes.txt   
│   ├── image_class_labels.txt
│   ├── train_test_split.txt
│   └── images.txt
├── data
│   ├── Bird_classes.pkl
│   ├── Bird_dictionary.pkl
│   └── Bird_glove6b_init_300d.npy
└── model
    └── model_9.pth # model name
``` 

# Prerequisites
- **`python 3.9`**
```python
pip install -r ./requirements.txt
```


# Project Description
- This project makes algorithm to make a good performance in fine-grained classification with CUB dates on basic backbone.
- It is based on https://arxiv.org/abs/2205.10529 paper's method

## Model Description
![image](https://user-images.githubusercontent.com/70640776/209456611-9efe5196-1f7a-452a-92e8-1215be9079d1.png)

- Input : (batchsize, 3, 448, 448) image 
- Output : (batchsize, 200) array (but when you test which sample image is in which class, you can ony get a class number)


## Dataset
- The dataset is available at http://www.vision.caltech.edu/datasets/
- Input: birds Image
- Output: 200 size array, you can get which class of bird it is 
- we only use the image of birds, not a bounding box or different information


## Example of train & test method

1. train 
- you can train the model with condition that you want
 ```python
python 3 main.py -m train -k 3
```
2. test
- you can test the model with sample image which class of bird it is 
 ```python
python 3 main.py -m test -md './CUB_200_2011/images/' -sd './CUB_200_2011/images/144.Common_Tern/Common_Tern_0078_149161.jpg' 
```
3. test_per_class_topk_k
- you can test top_k accuracy of each class
 ```python
python 3 main.py -m test_perclass -md './CUB_200_2011/images/' -sd './CUB_200_2011/images/144.Common_Tern/Common_Tern_0078_149161.jpg' -k 3
```

## Demo Image
![스크린샷 2022-12-25 오후 4 29 10](https://user-images.githubusercontent.com/70640776/209461633-8ea00b1c-60b3-4f51-a07e-2e0e7685f2bf.png)

## Reference
1. Fine-Grained Visual Classification using Self Assessment Classifier(https://arxiv.org/abs/2205.10529)
