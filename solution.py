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
import cv2
import torch
from action_invariance import ImageTransformer, TrimWrapper
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
        # Begin of dynamics and image transform code                                                                   #
        ################################################################################################################
        # Transformer for image invariance
        self.img_transformer = ImageTransformer()

        # Vars needed for trim estimation
        self.last_img = None
        self.current_img_trim = None
        self.log_ = []
        self.obs_counter = 0
        self.update_countdown = 20
        self.trim_wrapper = TrimWrapper()
        self.new_phi = None
        self.old_phi = None
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
        # Save image for trim estimation
        self.current_img_trim = obs

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
        self.old_phi = self.new_phi if self.new_phi is not None else None
        self.new_phi = pose[1]
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

        ################################################################################################################
        # Begin of trim wrapper code                                                                                   #
        ################################################################################################################
        if self.last_img is not None:
            delta_phi = self.new_phi - self.old_phi

            # Ignore first frames as the duckiebot is speeding up
            if self.obs_counter > 10:
                dphi = delta_phi.item()
                self.log_.append([dphi, pwm_left, pwm_right])
                self.update_countdown -= 1
                if not self.update_countdown:
                    self.trim_est = self.trim_wrapper.estimate_trim(self.log_)
                    self.update_countdown = 20

        pwm_left, pwm_right = self.trim_wrapper.undistort_action(pwm_left, pwm_right)
        self.last_img = self.current_img_trim
        ################################################################################################################
        ################################################################################################################

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
