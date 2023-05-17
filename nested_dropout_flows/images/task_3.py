import argparse
import json
from pathlib import Path

import numpy as np
import torch
from nflows import distributions, flows
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, SubsetRandomSampler

from data import create_dataset
from transform import create_transform

import pandas as pd


from bm3d import bm3d

import random


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
    n_classes = 10
    class_names = np.arange(n_classes)
    for i in range(n_classes):
      # Load validation data
      class_index = i
      eval_dataset = create_dataset(root=data_dir,
                                      split='test',
                                      class_index = class_index,
                                      **d_cf)

      c, h, w = eval_dataset.img_shape
      eval_image = torch.stack([eval_dataset.dataset[i][0] for i in range(2)], dim=0)
      # Denoise images
      imgs_den, regs, model_res, model_std = denoise(flow, eval_image, use_gpu, 300, 0.5, save_image_path=None)
      
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
    df = pd.DataFrame(m_res, columns=regs, index=class_names)
    df = df.transpose()
    df = df.round(1)
    df.to_csv('performance_results.csv')

    df_2 = pd.DataFrame(m_res_std, columns=regs, index=class_names)
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
    valid_indices = torch.load(run_dir / 'valid_indices.pt')
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
    flow_ckpt = run_dir / 'latest_flow.pt'
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

def denoise(flow, image, use_gpu, num_steps, step_size, save_image_path=None):
    
    device = torch.device('cuda')

    flow = flow.to(device)

    n_test = image.shape[0]
    n = image.shape[1]*image.shape[2]*image.shape[3]
       
    # Convert image to tensor and make a copy
    image = torch.tensor(image, dtype=torch.float32).to(device)
    image = image.clone().to(device)

    # Generate noise
    noise_type = "gaussian"

    if noise_type == "sinusoidal":
      x, y = np.meshgrid(np.linspace(0, 1, image.shape[2]), np.linspace(0, 1, image.shape[3]))
      row_variances = 10*np.sin(np.arange(height))
      noise = 10*np.sin(2*3*x + row_variances.reshape((-1, 1)) * y)
    elif noise_type == "poisson":
      lam = 20
      noise = np.random.poisson(lam, size=image.shape)
    else:
      noise_std = 10
      noise = np.random.normal(0, noise_std, size=(image.shape))

    
    noise_std = np.std(noise)
    print(noise_std)
    noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=device)
    x_noisy = image + noise

    # Initialize image
    z_sampled = np.random.normal(0, noise_std, [n_test, n])
    z_sampled = torch.tensor(z_sampled, requires_grad=True, dtype=torch.float, device=device)
    
    # Define the optimizer
    optimizer = torch.optim.Adam([z_sampled], lr=step_size)

    denoised_images = []
    denoised_images.append(image[0])
    denoised_images.append(x_noisy[0])

    regs = ["bm3d", "log", "robust_log", "l2", "robust_l2"]
    regs = ["l2", "robust_l2"]
    gamma_l2 = 90
    gamma_log = 40
    delta_l2 = 90/32
    delta_log = 40/32
    model_res = []
    model_std = []

    for reg in regs:

        z_sampled = np.random.normal(0, noise_std, [n_test, n])
        z_sampled = torch.tensor(z_sampled, requires_grad=True, dtype=torch.float, device=device)
        optimizer = torch.optim.Adam([z_sampled], lr=step_size)

        if reg == "bm3d":

            denoised_images_2 = []

            mse_loss = []
            for i in range(x_noisy.shape[0]):
                img = x_noisy[i].cpu().numpy().squeeze()
                img_denoised = bm3d(img, noise_std)
                img_denoised = torch.tensor(img_denoised[np.newaxis, :], dtype=torch.float32).to(device)
                psnr_t = torch.nn.MSELoss().to(device)
                psnr = psnr_t(image[i], img_denoised)
                mse_loss.append(psnr.item())
                denoised_images_2.append(img_denoised.unsqueeze(0))

            # Compute the average MSE loss over the entire batch
            mse_loss_2 = np.sum(mse_loss) / x_noisy.size(0)
            model_res.append(mse_loss_2)
            model_std.append(np.std(mse_loss))
            print("bm3d")
            print(mse_loss_2)
            print(np.std(mse_loss))
            print("")

            # Concatenate the denoised images along the batch dimension to create a new tensor
            denoised_images_2 = torch.cat(denoised_images_2, dim=0)
            denoised_images.append(denoised_images_2[0])
     
        else:
    
          for i in range(num_steps):
              # Zero the gradients
              optimizer.zero_grad()
              
              # Compute the loss as the negative log-likelihood of the denoised image
              x_gen = flow._transform.inverse(z_sampled)[0].to(device)
              
              x_gen_flat = x_gen.view([-1,n])
              # y_gen = torch.matmul(x_gen_flat, A)
              # x_gen = z_sampled
              x_gen = torch.reshape(x_gen, x_noisy.shape)
              residual_t  = ((x_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()
              if reg == "log":
                logabsdet = flow.log_prob(x_gen)
                residual_t -= gamma_log*(logabsdet).mean()
              if reg == "robust_log":
                logabsdet = flow.log_prob(x_gen)
                residual_t -= gamma_log*(logabsdet).mean()
                residual_t += delta_log*(x_gen.norm(dim=1, p=2)**2).mean()
              elif reg == "l2":
                residual_t += gamma_l2*(z_sampled.norm(dim=1, p=2)**2).mean()
              elif reg == "robust_l2":
                residual_t += delta_l2*(x_gen.norm(dim=1, p=2)**2).mean()
                residual_t += gamma_l2*(z_sampled.norm(dim=1, p=2)**2).mean()
                
              

              # psnr
          
              # psnr = 10 * np.log10(1 / psnr.item())
              if i>=1 and i%(num_steps-1)==0:
                psnr_t = torch.nn.MSELoss().to(device)
                mse_loss_orig = psnr_t(image, x_gen)
                mse_loss_2 = psnr_t(image, x_gen)
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


