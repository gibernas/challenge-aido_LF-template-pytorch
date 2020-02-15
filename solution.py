#!/usr/bin/env python3
import numpy as np
import time
from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage, logger

from submissionModel import Model
from wrappers import DTPytorchWrapper, SteeringToWheelVelWrapper
from PIL import Image
import io

########################################################################################################################
# Begin of image transform code                                                                                        #
########################################################################################################################
import torch
from action_invariance import ImageTransformer
########################################################################################################################

from controller import Controller


class PytorchAgent:
    def __init__(self, load_model=False, model_path=None):
        logger.info('PytorchAgent init')
        self.preprocessor = DTPytorchWrapper()

        self.model = Model()
        self.current_image = np.zeros((640, 480, 3))

        self.steering_to_wheel_wrapper = SteeringToWheelVelWrapper()

        self.controller = Controller()
        self.dt = None
        self.last_t = None
        self.old_obs = None

        ################################################################################################################
        # Begin of image transform code                                                                                #
        ################################################################################################################
        self.img_transformer = ImageTransformer()
        ################################################################################################################

        logger.info('PytorchAgent init complete')

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)

        ################################################################################################################
        # Begin of image transform code                                                                                #
        ################################################################################################################
        # Transform the observation
        obs = Image.fromarray(obs, mode='RGB')
        with torch.no_grad():
            obs = self.img_transformer.transform_img(obs)
        ################################################################################################################

        # self.current_image = self.preprocessor.preprocess(obs)
        self.current_image = obs

    def compute_action(self, observation):
        pose = self.model.predict(observation).detach().cpu().numpy()[0]
        pose[1] *= 3.1415
        time_now = time.time()
        if self.last_t is not None:
            self.dt = time_now - self.last_t
        v, omega = self.controller.compute_control_action(pose[0], pose[1], dt=self.dt)
        action = self.steering_to_wheel_wrapper.convert(np.array([v, omega]))
        self.last_t = time_now
        self.old_obs = observation
        return action.astype(float)

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)

        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = PytorchAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
