class Controller():
    def __init__(self, parameters):
        self.parameters = {
            '~k_d': -3.5,
            '~k_theta': -1,
            '~k_Id': 1,
            '~k_Iphi': 0,
            '~v_bar': 0.22,
            'd_resolution': 0.05,
            '~omega_to_rad_per_s': 4.75,
            'integral_bounds': {
                'phi': 1.2,
                'd': 0.3
            }
        }
        self.d_err = 0
        self.phi_err = 0
        self.d_I = 0
        self.phi_I = 0
        self.prev_d_err = 0
        self.prev_phi_err = 0

    def compute_control_action(self, d_err, phi_err, dt):

        if dt is not None:
            self.integrate_errors(d_err, phi_err, dt)

        self.d_I = self.adjust_integral(d_err, self.d_I, self.parameters['integral_bounds']['d'],
                                        self.parameters['d_resolution'])
        self.phi_I = self.adjust_integral(phi_err, self.phi_I, self.parameters['integral_bounds']['phi'],
                                          self.parameters['phi_resolution'])

        # Scale the parameters linear such that their real value is at 0.22m/s
        omega = self.parameters['~k_d'] * (0.22 / self.parameters['~v_bar']) * self.d_err + \
            self.parameters['~k_theta'] * (0.22 / self.parameters['~v_bar']) * self.phi_err

        omega -= self.parameters['~k_Id'] * (0.22/self.parameters['~v_bar']) * self.d_I
        omega -= self.parameters['~k_Iphi'] * (0.22/self.parameters['~v_bar']) * self.phi_I

        # apply magic conversion factors
        omega = omega * self.parameters['~omega_to_rad_per_s']

        self.prev_d_err = d_err
        self.prev_phi_err = phi_err

        return omega

    def integrate_errors(self, d_err, phi_err, dt):
        self.d_I += d_err * dt
        self.phi_I += phi_err * dt

    @staticmethod
    def adjust_integral(error, integral, bounds, resolution):
        if integral > bounds['top']:
            integral = bounds['top']
        elif integral < bounds['bot']:
            integral = bounds['bot']
        elif abs(error) < resolution:
            integral = 0
        return integral
