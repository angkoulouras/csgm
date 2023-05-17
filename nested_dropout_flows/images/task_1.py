import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.utils.validation import joblib
import torch
from nflows import distributions, flows
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, SubsetRandomSampler

from data import create_dataset
from transform import create_transform

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from skimage import io
from scipy.fftpack import dct, idct

import scipy.fftpack as fftpack
from bm3d import bm3d

import random

def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T, norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T, norm='ortho')


def vec(channels):
    image = np.zeros((32, 32, 1))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector):
    image = np.reshape(vector, [32, 32, 1])
    channels = [image[:, :, i] for i in range(1)]
    return channels


def sample(flow, eval_dataset, output_dir, use_gpu, seed, d_cf, run_dir, data_dir):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    grid = None

    res = []
    m_res = []
    m_res_std = []
    n_adv = 7
    noise_adv = [0, 2, 4, 6, 8, 10, 12]
    for i in range(n_adv):
      # Load validation data
      class_index = 6
      eval_dataset = create_dataset(root=data_dir,
                                      split='test',
                                      class_index = class_index,
                                      **d_cf)

      c, h, w = eval_dataset.img_shape
      eval_image = torch.stack([eval_dataset.dataset[i][0] for i in range(100)], dim=0)
      # Denoise images
      imgs_den, regs, model_res, model_std = denoise(flow, eval_image, use_gpu, 300, 0.5, noise_adv[i], save_image_path=None)
      
      print("")
      print(regs)
      print(model_res)
      stacked_imgs = torch.stack([eval_dataset.preprocess_fn.inverse(img_den)[0] for img_den in imgs_den], dim=0)
      grid_new = make_grid(stacked_imgs, nrow=1, padding=0)
      grid_new = torch.reshape(grid_new, [1, len(imgs_den)*32, 32])
      # print(grid_new.shape)

      if grid is None:
          grid = grid_new
          res.append(regs)
      else:
          grid = torch.cat([grid, grid_new], dim=2)

      m_res.append(model_res)
      m_res_std.append(model_std)

    # save the grid as a PNG image
    save_image(grid, 'combined.png', normalize=True)

    # create a DataFrame and save the DataFrame to a CSV file
    df = pd.DataFrame(m_res, columns=regs, index=noise_adv)
    df = df.transpose()
    df = df.round(1)
    df.to_csv('performance_results.csv')

    df_2 = pd.DataFrame(m_res_std, columns=regs, index=noise_adv)
    df_2 = df_2.transpose()
    df_2 = df_2.round(1)
    df_2.to_csv('performance_results_std.csv')

    flow = flow.to(device)
    flow.eval()

    # Disable gradients for the rest of the run
    torch.set_grad_enabled(False)

    # Samples
    images = flow.sample(18)
    images = eval_dataset.preprocess_fn.inverse(images)
    save_image(images, output_dir / 'samples.pdf', nrow=6)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=314833845)
    parser.add_argument('--class_index', type=int, default=0)

    script_dir = Path(__file__).resolve().parent
    parser.add_argument('--data_config', type=str,
                        default=script_dir / 'config' / 'data_config.json')
    parser.add_argument('--flow_config', type=str,
                        default=script_dir / 'config' / 'flow_config.json')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)

    print('Loading data')

    with open(args.data_config) as fp:
        data_config = json.load(fp)

    # Load validation data
    valid_indices = torch.load(run_dir / 'valid_indices_class_6.pt')
    eval_dataset = create_dataset(root=args.data_dir,
                                    split='test',
                                    class_index = args.class_index,
                                    **data_config)

    print('Creating a flow')

    with open(args.flow_config) as fp:
        flow_config = json.load(fp)

    c, h, w = eval_dataset.img_shape
    distribution = distributions.StandardNormal((c * h * w,))
    transform = create_transform(c, h, w,
                                 num_bits=data_config['num_bits'],
                                 **flow_config)
    flow = flows.Flow(transform, distribution)

    # Load checkpoint
    flow_ckpt = run_dir / 'latest_flow_class_6.pt'
    flow.load_state_dict(torch.load(flow_ckpt))
    print(f'Flow checkpoint loaded: {flow_ckpt}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print('Sampling')
    sample(flow=flow,
           eval_dataset=eval_dataset,
           output_dir=output_dir,
           use_gpu=args.use_gpu,
           seed=args.seed,
           d_cf = data_config, run_dir = run_dir, data_dir=args.data_dir)

def denoise(flow, image, use_gpu, num_steps, step_size, noise_coeff, save_image_path=None):
    
    device = torch.device('cuda')

    flow = flow.to(device)

    n_test = image.shape[0]
    n = image.shape[1]*image.shape[2]*image.shape[3]
       
    # Convert image to tensor and make a copy
    image = torch.tensor(image, dtype=torch.float32).to(device)
    image = image.clone().to(device)

    # Compressed sensing 
    m = n // 10

    # Generate noise
    noise_std = 1
    noise = np.random.normal(0, noise_std, size=(m))
    noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=device)
    noise = noise * 0.1 / np.sqrt(m)

 
    delta = 2.0
    A = np.random.normal(0,1/np.sqrt(m), size=(n,m))
    E = np.random.normal(0, 1, A.shape)
    norm_E = np.linalg.norm(E, ord='fro')
    scaled_E = (noise_coeff * delta / norm_E) * E
    A_noisy = A + scaled_E
    A = torch.tensor(A,dtype=torch.float, requires_grad=False, device=device)
    A_noisy = torch.tensor(A_noisy,dtype=torch.float, requires_grad=False, device=device)
    image_flat = image.view([-1,n])
    noise_flat = noise.view([-1, m])
    x_noisy = torch.matmul(image_flat, A) + noise_flat

    # Initialize image
    z_sampled = np.random.normal(0, 1, [n_test, n])
    z_sampled = torch.tensor(z_sampled, requires_grad=True, dtype=torch.float, device=device)
    
    # Define the optimizer
    optimizer = torch.optim.Adam([z_sampled], lr=step_size)

    denoised_images = []
    denoised_images.append(image[0])
    # denoised_images.append(x_noisy[0])

    regs = ["lasso", "log", "robust_log", "l2", "robust_l2"]
    gamma_l2 = 80.0
    gamma_log = 15.0
    model_res = []
    model_std = []

    for reg in regs:

        z_sampled = np.random.normal(0, 1, [n_test, n])
        z_sampled = torch.tensor(z_sampled, requires_grad=True, dtype=torch.float, device=device)
        optimizer = torch.optim.Adam([z_sampled], lr=step_size)

        if reg == "lasso":
            
            mse_loss = []

            for i in range(image.shape[0]):

                A_lasso = A_noisy.cpu().numpy().copy()
                for j in range(A.shape[1]):
                    A_lasso[:, j] = vec([dct2(channel) for channel in devec(A_lasso[:, j])])

                # Solve the Lasso problem
                lasso = Lasso(alpha=0.00035, max_iter=5000)
                lasso.fit(A_lasso.T, (x_noisy[0, :]).cpu().numpy())
                x_sparse = lasso.coef_.reshape(1, -1)

                # Reconstruct the image
                x_hat = vec([idct2(channel) for channel in devec(x_sparse)]).T
                x_hat = x_hat.reshape(image[i].shape)
                img_denoised = torch.tensor(x_hat, dtype=torch.float32).to(device)
                psnr_t = torch.nn.MSELoss().to(device)
                psnr = psnr_t(image[i], img_denoised)
                mse_loss.append(psnr.item())

                if(i==0):
                  denoised_images.append(img_denoised)

            # Compute the average MSE loss over the entire batch
            mse_loss_2 = np.sum(mse_loss) / image.shape[0]
            model_res.append(mse_loss_2)
            model_std.append(np.std(mse_loss))
            print("lasso_dct")
            print(mse_loss_2)
            print(np.std(mse_loss))
            print("")

        else:
    
          for i in range(num_steps):
              # Zero the gradients
              optimizer.zero_grad()
              
              # Compute the loss as the negative log-likelihood of the denoised image
              x_gen = flow._transform.inverse(z_sampled)[0].to(device)
              x_gen_flat = x_gen.view([-1,n])
              y_gen = torch.matmul(x_gen_flat, A_noisy)
              y_gen = torch.reshape(y_gen, x_noisy.shape)

              residual_t  = ((y_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()
              if reg == "log":
                logabsdet = flow.log_prob(x_gen)
                residual_t -= gamma_log*(logabsdet).mean()
              if reg == "robust_log":
                logabsdet = flow.log_prob(x_gen)
                residual_t -= gamma_log*(logabsdet).mean()
                residual_t += noise_coeff*delta*(x_gen.norm(dim=1, p=2)**2).mean()
              elif reg == "l2":
                residual_t += gamma_l2*(z_sampled.norm(dim=1, p=2)**2).mean()
              elif reg == "robust_l2":
                residual_t += noise_coeff*(x_gen.norm(dim=1, p=2)**2).mean()
                residual_t += gamma_l2*(z_sampled.norm(dim=1, p=2)**2).mean()


              # psnr
          
              # psnr = 10 * np.log10(1 / psnr.item())
              if i>=1 and i%(num_steps-1)==0:
                psnr_t = torch.nn.MSELoss().to(device)
                mse_loss_2 = psnr_t(image_flat, x_gen_flat)
                mse_loss = []
                for i in range(x_noisy.shape[0]):
                  psnr = psnr_t(image[i], x_gen[i])
                  mse_loss.append(psnr.item())
                model_res.append(mse_loss_2.item())
                model_std.append(np.std(mse_loss))
                print(reg)
                print(mse_loss_2.item())
                print(np.std(mse_loss))
                print("")
              
              # Backpropagate the loss
              residual_t.backward()
              
              # Take a step in the opposite direction of the gradients
              optimizer.step()
              
              # # Clip the denoised image to the valid range of pixel values
              # denoised_image.data.clamp_(0, 1)
              
              # # Print the loss every 10 steps
              # if (i + 1) % 10 == 0:
              #     print(f"Step {i + 1}: Loss = {-log_likelihood.item():.6f}")
                  
              # # Save the denoised image every 100 steps
              # if save_image_path is not None and (i + 1) % 100 == 0:
              #     save_image(denoised_image.detach().cpu(), save_image_path)

          denoised_images.append(x_gen[0])
            
    # Return the denoised image as a numpy array

    return denoised_images, regs, model_res, model_std

if __name__ == '__main__':
    main()
