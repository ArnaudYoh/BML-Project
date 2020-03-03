"""Classes for defining our training environments"""

import numpy as np


class SimpleDemo:
    """Basic Bimodal Distribution problem class"""

    def __init__(self):
        self._theta1 = 0
        self._theta2 = 1
        self.sigx = 2

    def get_data(self, n_samples=100):
        r"""
        Gets n data points from a bimodal distribution

        The distribution is defined as:
        .. math::
            \frac{1}{2} N(\theta_1, \sigma^2_x) +
            \frac{1}{2} N(\theta_1 + \thetat_2, \sigma_x^2)

        Args:
            n_samples (int): Number of samples

        Returns:
            list: list of the sampled floats
        """

        data = np.zeros(n_samples)

        for i in range(n_samples):
            data[i] = 1/2 * np.random.normal(self._theta1, self.sigx) + \
                      1/2*np.random.normal(self._theta1, self.sigx)

        return data


class LogReg:
    """Logistic Regression problem class"""

    def __init__(self):
        self._theta1 = 0
        self._theta2 = 1
        self.sigx = 2

    def get_data(self, n_samples=100):
        r"""
        Gets n data points from the a9a dataset (Lin. et al, 2008)
        """
        pass

class ICA:
    """Independent Components Analysis problem class"""

    def __init__(self):
        self._theta1 = 0
        self._theta2 = 1
        self.sigx = 2

    def get_data(self, n_samples=1000):
        r"""
        Gets n data points from the a9a dataset (Lin. et al, 2008)
        """
        pass