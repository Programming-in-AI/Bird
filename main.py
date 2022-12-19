from train import train, set_device
from test import test, test_per_class, test_per_class_topk
import random

import torch

# random.seed(42)
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def main():
    # gonna implement parse arguement train test

    device = set_device()
    # # train
    img_folder_dir = './CUB_200_2011/images/'
    # train(root_dir=img_folder_dir, device=device)

    # test
    input_img_path = '/Users/dongwook/Desktop/Project/hw_beida/AI/CUB_200_2011/images/144.Common_Tern/Common_Tern_0078_149161.jpg'
    model_path = '/Users/dongwook/Desktop/Project/hw_beida/AI/models/model_9.pth'
    # test(model_path, input_img_path)
    _, wrong_index = test_per_class_topk(root_dir=img_folder_dir, model_path=model_path, device=device, top_k= 1)
    #_, wrong_index = test_per_class(root_dir=img_folder_dir, model_path=model_path, device=device)
    # class_acc = test_per_class1(root_dir=img_folder_dir, model_path=model_path, device=device)


if __name__ == '__main__':
    main()

