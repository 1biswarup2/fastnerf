#%%
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary
import os 
from torch.utils.tensorboard import SummaryWriter
import sys
import gc
import argparse
from itertools import chain
import atexit
            
class FastNerf(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_direction=4, hidden_dim_pos=384, hidden_dim_dir=128, D=8):
        super(FastNerf, self).__init__()

        self.Fpos = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, hidden_dim_pos), nn.ReLU(),
                                  nn.Linear(hidden_dim_pos, 3 * D + 1), )

        self.Fdir = nn.Sequential(nn.Linear(embedding_dim_direction * 6 + 3, hidden_dim_dir), nn.ReLU(),
                                  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
                                  nn.Linear(hidden_dim_dir, hidden_dim_dir), nn.ReLU(),
                                  nn.Linear(hidden_dim_dir, D), )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.D = D

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        # print(out[0].shape,"OUT SHAPE")
        
        print("x",x.shape)
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        # print(torch.cat(out, dim=1).size,"size")
        
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        # print(o.shape,d.shape)
        # print("POSITIONAL ENCODING")
        sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim_pos))
        # print("sigma_uvw",sigma_uvw.shape)
        
        
        sigma = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])  # [batch_size, 1]
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))  # [batch_size, 3, D]
        
        # print("POSITIONAL ENCODING 2")
        beta = torch.softmax(self.Fdir(self.positional_encoding(d, self.embedding_dim_direction)), -1)
        color = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]

        return color, sigma

def softmax_(x, dim):
        x_max = x.max(dim=dim, keepdim=True).values
        x.sub_(x_max).exp_().div_(x.sum(dim=dim, keepdim=True))
        
class Cache(nn.Module):
    def __init__(self, model, scale, device, Np, Nd):
        super(Cache, self).__init__()

        with torch.no_grad():
            # Position
            x, y, z = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device)])
            # print(x.shape,"x shape")
            # print(y.shape,"y shape")
            # print(z.shape,"z shape")
            xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
            # print(xyz.shape,"xyz shape")
            sigma_uvw = model.Fpos(model.positional_encoding(xyz, model.embedding_dim_pos))
            self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
            # Direction
            xd, yd = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Nd).to(device),
                                     torch.linspace(-scale / 2, scale / 2, Nd).to(device)])
            xyz_d = torch.cat((xd.reshape(-1, 1), yd.reshape(-1, 1),
                               torch.sqrt((1 - xd ** 2 - yd ** 2).clip(0, 1)).reshape(-1, 1)), dim=1)
            beta = model.Fdir(model.positional_encoding(xyz_d, model.embedding_dim_direction))
            self.beta = beta.reshape((Nd, Nd, -1))
            print ("Beta in cache " ,self.beta.shape)

        self.scale = scale
        self.Np = Np
        self.Nd = Nd
        self.D = model.D
    
    

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0], 1), device=x.device)

        print("x",x.shape)
        print("d",d.shape)

        mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (x[:, 2].abs() < (self.scale / 2))
        # mask is done to check if the x is within the range of scale/2 , any other method? to do the same thing
        # Position
        idx = (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1)
        print("indexed shape",self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]].shape)
        print("idx",idx.shape)
        print("self.sigma_uvw",self.sigma_uvw.shape)
        print("mask",mask.shape)
        sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]]
        print(self.beta[idx[:, 0], idx[:, 1]].shape)
        print("beta_indexed",idx[:, 0].shape,idx[:, 1].shape,idx[:, 2].shape)
        print("sigma_uvw",sigma_uvw.shape)
        print("beta",self.beta.shape)
        # Direction
        # idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1)
        beta = softmax_(self.beta[idx[:, 0], idx[:, 1]], -1)
        beta=self.beta[idx[:, 0], idx[:, 1]]
        # 
        # beta=torch.softmax(self.beta[idx[:1000, 0], idx[:1000, 1]], -1)

        sigma[mask] = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None]) 
        print("sigma",sigma.shape)
        #uvw = torch.sigmoid_(sigma_uvw[:, 1:])
        # print("uvw",uvw.shape)
        # print("uvw_reshaped" , uvw.reshape(-1, 3, self.D).shape)
        uvw = torch.sigmoid_(sigma_uvw[:, 1:].reshape(-1, 3, self.D))
        print("uvw",uvw.shape)
        uvw.mul_(beta.unsqueeze(1))
        print("uvw_multiplied",uvw.shape)
        color[mask] = uvw.sum(-1) 
        return color, sigma

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    print("t",t.shape)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    print("lower",lower.shape)
    print("upper",upper.shape)
    u = torch.rand(t.shape, device=ray_origins.device)
    
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    print("t",t.shape)
    
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)
    print("delta",delta.shape)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  # [batch_size, nb_bins, 3]
    print("x",x.shape)
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)
    print("ray_directions",ray_directions.shape)
    
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    print("colors",colors.shape)
    
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1) # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def test(model, hn, hf, dataset, img_index=0, nb_bins=192, H=400, W=400):
    # print("test:img_index:",img_index)
    
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    print("test:ray_origins:",ray_origins.shape)
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    print("test:ray_directions:",ray_directions.shape)
    
    regenerated_px_values = render_rays(model, ray_origins.to(device), ray_directions.to(device), hn=hn, hf=hf,
                                        nb_bins=nb_bins)
    
    
    
    print("test:regenerated_px_values:",regenerated_px_values.shape)
    # print("test:regenerated_px_values",regenerated_px_values.shape)
    # print("test:regenerated_px_values:1:",regenerated_px_values.data.cpu().numpy().reshape(H, W, 3).clip(0, 1).shape)

    fig = plt.figure()
    fig.set_size_inches(H, W)
    plt.imshow(regenerated_px_values.data.cpu().numpy().reshape(H, W, 3).clip(0, 1))
    plt.axis('off')
    plt.savefig(f'novel_views/img_test_{img_index}.png', dpi=1)
    print('Render successful{img_index}')
    # plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()

    # generated_px_values = dataset[img_index * H * W: (img_index + 1) * H * W, 6:]
    # fig=plt.figure()
    # fig.set_size_inches(H, W)
    # plt.imshow(generated_px_values.data.cpu().numpy().reshape(H, W, 3).clip(0, 1))
    # plt.axis('off')
    # plt.savefig(f'novel_views/img_generated_{img_index}.png', dpi=1)
    # print('Testing successful_{img_index}')

def cleanup():
    """
    Cleanup function to release GPU resources.
    """
    # Clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192):
    # print("dataloader", type(data_loader))
    training_loss = []
    for _ in (range(nb_epochs)):
        for ep, batch in enumerate(tqdm(data_loader)):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)
            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()
        torch.save(nerf_model.cpu(), 'nerf_model')
        nerf_model.to(device)
    return training_loss

def pattern(dataset_size):
    # Set the device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create sequence as PyTorch tensor
    seq = torch.arange(dataset_size, device=device)

    # Define steps and calculations
    step1 = 32
    step2 = 400
    step3 = step2 // 2
    subt = step1 // 2
    range_start = step3 - subt

    # First Transformation
    indices = torch.arange(range_start, seq.numel(), step2, device=device)
    result = torch.cat([seq[i:i+step1] for i in indices])

    # Second Transformation
    step11 = step1 * step1
    range_start1 = range_start * step1
    step21 = range_start1 * 2 + step11
    indices1 = torch.arange(range_start1, result.numel(), step21, device=device)
    result1 = torch.cat([result[i:i+step11] for i in indices1])

    # Converting result back to CPU for return, if needed
    return result1.cpu().numpy()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Neural Rendering with FastNeRF")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help="Mode to run the script: 'train' or 'test'")
    args = parser.parse_args()
    
    # images , labels = training_dataset
    # print(images.shape)
    # sys.exit()
    testing_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
    model = FastNerf().to(device)
    atexit.register(cleanup)

    if args.mode == 'train':
        training_dataset = torch.from_numpy(np.load('training_data.pkl', allow_pickle=True))
        training_dataset_shape = np.shape(training_dataset)
        # print("training_dataset_shape",training_dataset_shape)
        pattern1=pattern(training_dataset_shape[0])
        # print("pattern1",pattern1)
        training_dataset_new=training_dataset[pattern1,:]
        training_dataset_new_shape = np.shape(training_dataset_new)
        # print("training_dataset_new_shape",training_dataset_new_shape
        model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
        data_loader = DataLoader(training_dataset_new, batch_size=1024, shuffle=True)
        train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6)

        # Save the model
        model_path = 'model.pth'
        torch.save(model, model_path)
        model_size = os.path.getsize(model_path)
        print(f"Size of the saved model on disk: {model_size / (1024*1024)} MB")

    elif args.mode == 'test':
        model = torch.load('model.pth')
        # model.eval()
        cache = Cache(model, 2.2, 'cuda', 64, 64)
        print("cache")
        
        for idx in range(200):
            test(cache, 2., 6., testing_dataset, img_index=idx, nb_bins=12, H=400, W=400)
            torch.cuda.empty_cache()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()

# %%

