import math

import numpy as np

from ddpm.diffusion import generate_linear_schedule

if __name__ == '__main__':
    large = math.pow(10, -2)
    small = 2*math.pow(10,-4)
    betas = generate_linear_schedule(2000, small, large)
    print(betas)

    alphas = 1.0 - betas
    print(alphas)
    alphas_cumprod = np.cumprod(alphas)
    print(alphas_cumprod)

