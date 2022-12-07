import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def test(model_path, input_img_path):

    model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 200)

    # load model
    loaded_parameter = torch.load(model_path)
    model.load_state_dict(loaded_parameter, strict=False)

    # make input have batch channel
    tr = transforms.ToTensor()
    input_img = tr(Image.open(input_img_path))
    input_img = torch.unsqueeze(input_img, 0)

    # model
    with torch.no_grad():
        model.eval()
        output = model(input_img)

    # output should be tensor type
    print(f'This image is {torch.argmax(output).item()}th class')