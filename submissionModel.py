import torch
from torchvision import transforms
from PIL import Image

from utils import TransCropHorizon
from models import VanillaCNN
device = torch.device("cpu")
from aido_schemas import  logger


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        image_res = 64

        self.transformsCustom = transforms.Compose([transforms.Resize(image_res),
                                                    TransCropHorizon(0.5, set_black=False),
                                                    transforms.Grayscale(num_output_channels=1),
                                                    transforms.ToTensor()])

        model_name = 'VanillaCNN_1579009324.1649516_lr_0.0001_bs_16_dataset_sim_totepo_200_epo199_final.pt'
        model_path = '/'.join(['models', model_name])
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.double().to(device=device)

    def close(self):
        # TODO: release resources
        pass

    def predict(self, images):
        images = self.transformsCustom(Image.fromarray(images.astype('uint8').T))
        images = images.double().to(device=device)
        images = images.unsqueeze(1)
        pose_estimates = self.model(images)

        return pose_estimates
