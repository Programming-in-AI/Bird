from train import train
from train_utils import set_device
from test import test, test_per_class, test_per_class_topk
import random
import torch
import argparse
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='train or test or test per class')
    parser.add_argument('-m', '--mode', dest="mode", help='select mode which is train or test or test per class')
    parser.add_argument('-md', '--model_directory', dest="model_directory", default='./models/model_9.pth', help='model_path')
    parser.add_argument('-sd', '--sample_directory ', dest="sample_directory",default='./CUB_200_2011/images/144.Common_Tern/Common_Tern_0078_149161.jpg', help='sample image directory')
    parser.add_argument('-k', '--top_k ', dest="top_k", default=1, help='tune top k vale when testing model ')

    args = parser.parse_args()
    # img_folder_dir = './CUB_200_2011/images/'
    # sample_img_path = './CUB_200_2011/images/144.Common_Tern/Common_Tern_0078_149161.jpg'
    # model_path = './models/model_9.pth'

    img_folder_dir = args.mode
    sample_img_path = args.sample_directory
    model_path = args.model_directory
    top_k = args.top_k


    if args.mode == 'train':
        train(root_dir=img_folder_dir, device=set_device(), top_k= top_k)
    elif args.mode == 'test':
        test(model_path, input_img_path=sample_img_path)
    elif args.mode == 'test_perclass':
        test_per_class_topk(root_dir=img_folder_dir, model_path=model_path, device=set_device(), top_k=1)


if __name__ == '__main__':
    main()

