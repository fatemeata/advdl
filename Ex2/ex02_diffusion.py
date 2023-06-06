import math

import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    t = t/timesteps
    f = torch.cos(((t + s)/(1+s))*math.pi*0.5) ** 2
    alphas = f/f[0]
    betas = 1 - (alphas[1:]/alphas[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas

def sigmoid_beta_schedule(beta_start, beta_end, timesteps, clip_min=1e-9):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the
    #  sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    t = 2*t/timesteps
    v_start = torch.tensor(beta_start).sigmoid()
    v_end = torch.tensor(beta_end).sigmoid()
    f = ((t * (beta_end - beta_start) + (beta_start-beta_end))).sigmoid()
    f =  (v_end - f) / (v_end - v_start)
    alphas = f/f[0]
    betas = 1 - (alphas[1:]/alphas[:-1])
    betas = torch.clip(betas, clip_min, 0.999)
    return betas

class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want
    #  to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly
        #  use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        # define alphas
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t.
        #  Note that x and t both have a batch dimension

        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model.predict(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        # TODO (2.2): The method should return the image at timestep t-1.
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            image_t_1 = model_mean + torch.sqrt(posterior_variance_t) * noise
            return image_t_1

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing''
        #  the noise in the generated image.
        device = next(model.parameters()).device
        image = torch.randn((batch_size, channels, image_size, image_size), device=device)
        images = [image]

        for t in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            image = self.p_sample(model, image, torch.full((batch_size,), t, device=device, dtype=torch.long), t)
            images.append(image)

        # TODO (2.2): Return the generated images
        return images

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor;
        #  if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape
        )
        noisy_image = sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise

        return noisy_image

    def p_losses(self, denoise_model, x_zero, t, classes, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using
        #  the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        x_noisy = self.q_sample(x_zero=x_zero, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t, classes=classes)
        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
