import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from dataloader import CustomDataset, Dataloader, read_txt
import pandas as pd

def load_model(model_path):
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 200)

    loaded_parameter = torch.load(model_path)
    model.load_state_dict(loaded_parameter, strict=False)
    return model


def test(model_path, input_img_path):

    # load model
    model = load_model(model_path)

    # make input have batch channel
    tr = transforms.ToTensor()
    input_img = tr(Image.open(input_img_path))
    input_img = torch.unsqueeze(input_img, 0)

    # model
    with torch.no_grad():
        model.eval()
        output = model(input_img)

    # output should be tensor type
    class_num = torch.argmax(output).item()
    print(f'This image is {class_num}th class')

    return class_num


def test_per_class(root_dir, model_path, device):
    print('[Test dataset Processing...]')
    dataset = CustomDataset(root_dir, isTrain=True)
    print("Training data size : {}".format(dataset.__len__()[0]))
    print("Validating data size : {}".format(dataset.__len__()[1]))
    batch_size = 32
    val_dataloader = Dataloader(dataset.val_dataset, batch_size)

    model = load_model(model_path)
    class_acc = np.zeros(200, dtype=float)
    right = np.zeros(200, dtype=int)
    wrong = np.zeros(200, dtype=int)
    wrong_index = np.array([])

    for i, (img, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        model = model.to(device)
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            h = model(img)
            _, y_pred = h.max(1)

        ans_list = (label == y_pred).float()
        
        for j in range(len(ans_list)):
            if ans_list[j] == 1:
                right[label[j]] += 1
            else:
                wrong[label[j]] += 1
                np.append(wrong_index, i*len(label) + j)

    # accuracy per class
    for i in range(len(class_acc)):
        class_acc[i] = round(right[i] / (right[i] + wrong[i]), 2)

    # make dataframe and save in txt
    df = pd.DataFrame({'class': [], 'acc': []})
    for i, item in enumerate(class_acc):
        df.loc[i] = [str(i) + 'th', str(item) + '%']
    df.to_csv('acc_per_class.txt', sep='\t', index=False)

    line = read_txt('./CUB_200_2011/train_test_split.txt')

    val_sequence = np.array([int(item.split(' ')[0]) for item in line if line.split(' ')[1] == '0'])

    return class_acc, val_sequence[wrong_index]


def test_per_class_topk(root_dir, model_path, device, top_k):
    print('[Test dataset Processing...]')
    dataset = CustomDataset(root_dir, isTrain=True)
    print("Training data size : {}".format(dataset.__len__()[0]))
    print("Validating data size : {}".format(dataset.__len__()[1]))
    batch_size = 16
    val_dataloader = Dataloader(dataset.val_dataset, batch_size)

    model = load_model(model_path)
    class_acc = np.zeros(200, dtype=float)
    right = np.zeros(200, dtype=int)
    wrong = np.zeros(200, dtype=int)
    wrong_index = np.array([])
    for i, (img, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        model = model.to(device)
        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            h = model(img)
            _, y_pred = torch.topk(h, k=top_k, dim=1)

        ans_list = ([1 if label[i] in y_pred[i] else 0 for i in range(len(y_pred))])

        for j in range(len(ans_list)):
            if ans_list[j] == 1:
                right[label[j]] += 1
            else:
                wrong[label[j]] += 1
                np.append(wrong_index, i * len(label) + j)

    # accuracy per class
    for i in range(len(class_acc)):
        class_acc[i] = round(right[i] / (right[i] + wrong[i]) * 100, 2)

    # make dataframe and save in txt
    df = pd.DataFrame({'class': [], 'acc': []})
    for i, item in enumerate(class_acc):
        df.loc[i] = [str(i) + 'th', str(item) + '%']
    df.to_csv('topk_'+str(top_k)+'_acc_per_class.txt', sep='\t', index=False)

    line = read_txt('./CUB_200_2011/train_test_split.txt')

    val_sequence = np.array([int(item.split(' ')[0]) for item in line if line.split(' ')[1] == '0'])

    return class_acc, val_sequence[wrong_index]
