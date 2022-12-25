
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
│   │   ├── 00000.npz
│   │   ├── ***
│   │   └── 03983.npz
│   ├── parts
│   │  └── model_27.pth
│   ├── attributes
│   │   ├── Renju1
│   │   ├── Renju2
│   │   └── Renju3
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
- Output : (batchsize, 200) array
(output is the class number that model predict)


## Dataset
- The dataset is available at http://www.vision.caltech.edu/datasets/
- Input: birds Image
- Output: 200 size array, you can get which class of bird it is 
- we only use the image of birds, not a bounding box or different information


## train & test method

1. Use a 15*15 grid board
2. Randomly assign the black and white player
3. Black always begins
4. The black player's first stone must be placed in the center of the board
5. If any player wins twice, the whole game is over
6. Blocking samsam which is the strategy that black cannot use is not implemented yet
7. The second black stone cannot be placed inside 5x5 center area (since Black should be penalized for being able to place the first stone)

## Demo Image
![스크린샷 2022-12-25 오후 4 29 10](https://user-images.githubusercontent.com/70640776/209461633-8ea00b1c-60b3-4f51-a07e-2e0e7685f2bf.png)

## Reference
1. Fine-Grained Visual Classification using Self Assessment Classifier(https://arxiv.org/abs/2205.10529)
