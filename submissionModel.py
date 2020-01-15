import torch
from torchvision import transforms
from PIL import Image

from utils.utils import TransCropHorizon

device = torch.device("cpu")


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        image_res = 64

        self.transformsCustom = transforms.Compose([transforms.Resize(image_res),
                                                    TransCropHorizon(0.5, set_black=False),
                                                    transforms.Grayscale(num_output_channels=1),
                                                    transforms.ToTensor()])

        self.model = torch.load('whatevermodel', map_location=device)
        self.model.double().to(device=device)

    def close(self):
        # TODO: release resources
        pass

    def predict(self, images):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        images = self.transformsCustom(Image.fromarray(images))
        images = images.double().to(device=device)
        images = images.unsqueeze(1)
        pose_estimates = self.model(images)
        return pose_estimates
