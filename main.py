from train import train
from test import test
from vit_pytorch import ViT


def main():
    # train
    img_folder_dir = './CUB_200_2011/images/'
    train(root_dir=img_folder_dir)

    # test
    input_img_path = '/Users/dongwook/Desktop/Project/hw_beida/AI/CUB_200_2011/images/' \
                 '200.Common_Yellowthroat/Common_Yellowthroat_0070_190678.jpg'
    model_path = '/Users/dongwook/Desktop/Project/hw_beida/AI/models/model_8.pth'
    test(model_path, input_img_path)


if __name__ == '__main__':
    main()

