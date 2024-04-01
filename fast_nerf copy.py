import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

torch.cuda.set_per_process_memory_fraction(0.5, device=0)


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
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        sigma_uvw = self.Fpos(self.positional_encoding(o, self.embedding_dim_pos)) #s
        sigma = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None]) 
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))

        beta = torch.softmax(self.Fdir(self.positional_encoding(d, self.embedding_dim_direction)), -1)
        color = (beta.unsqueeze(1) * uvw).sum(-1)  # calculation of color of each pixel in each ray
        return color, sigma


class Cache(nn.Module):
    def __init__(self, model, scale, device, Np, Nd):
        super(Cache, self).__init__()

        with torch.no_grad():
            # Position
            x, y, z = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device),
                                      torch.linspace(-scale / 2, scale / 2, Np).to(device)])
            xyz = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), dim=1)
            sigma_uvw = model.Fpos(model.positional_encoding(xyz, model.embedding_dim_pos))
            self.sigma_uvw = sigma_uvw.reshape((Np, Np, Np, -1))
            # Direction
            xd, yd = torch.meshgrid([torch.linspace(-scale / 2, scale / 2, Nd).to(device),
                                     torch.linspace(-scale / 2, scale / 2, Nd).to(device)])
            xyz_d = torch.cat((xd.reshape(-1, 1), yd.reshape(-1, 1),
                               torch.sqrt((1 - xd ** 2 - yd ** 2).clip(0, 1)).reshape(-1, 1)), dim=1)
            beta = model.Fdir(model.positional_encoding(xyz_d, model.embedding_dim_direction))
            self.beta = beta.reshape((Nd, Nd, -1))

        self.scale = scale
        self.Np = Np
        self.Nd = Nd
        self.D = model.D
        torch.cuda.empty_cache()

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0], 1), device=x.device)
        mask = (x[:, 0].abs() < (self.scale / 2)) & (x[:, 1].abs() < (self.scale / 2)) & (x[:, 2].abs() < (self.scale / 2)) # mask for valid rays
        # Position
        idx = (x[mask] / (self.scale / self.Np) + self.Np / 2).long().clip(0, self.Np - 1) # voxel index 
        sigma_uvw = self.sigma_uvw[idx[:, 0], idx[:, 1], idx[:, 2]] 
        # Direction
        idx = (d[mask] * self.Nd).long().clip(0, self.Nd - 1) 
        beta = torch.softmax(self.beta[idx[:, 0], idx[:, 1]], -1)
        sigma[mask] = torch.nn.functional.softplus(sigma_uvw[:, 0][..., None])  # [batch_size, 1]
        uvw = torch.sigmoid(sigma_uvw[:, 1:].reshape(-1, 3, self.D))  # [batch_size, 3, D]
        color[mask] = (beta.unsqueeze(1) * uvw).sum(-1)  # [batch_size, 3]
        return color, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=150):
    device = ray_origins.device
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2. # Midpoint example in detail , t[:, :-1] is the first 191 elements of t, t[:, 1:] is the last 191 elements of t example : t = [1,2,3,4,5] t[:, :-1] = [1,2,3,4] t[:, 1:] = [2,3,4,5] so mid = [1.5,2.5,3.5,4.5]
    lower = torch.cat((t[:, :1], mid), -1) # lower is the first 191 elements of t 
    upper = torch.cat((mid, t[:, -1:]), -1) # upper is the last 191 elements of t
    u = torch.rand(t.shape, device=ray_origins.device) 
    t = lower + (upper - lower) * u  # Perturb sampling along each ray.
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3)) # color = [batch_size*nb_bins, 3] sigma = [batch_size*nb_bins, 1]
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta) 
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  # Pixel values
    weight_sum = weights.sum(-1).sum(-1) # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)


@torch.no_grad()
def test(model, hn, hf, dataset, img_index=0, nb_bins=192, H=400, W=400):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    regenerated_px_values = render_rays(model, ray_origins.to(device), ray_directions.to(device), hn=hn, hf=hf,
                                        nb_bins=nb_bins)
    plt.figure()
    plt.imshow(regenerated_px_values.data.numpy().reshape(H, W, 3).clip(0, 1))
    plt.axis('off')
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()
    torch.cuda.empty_cache()


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192):
    training_loss = []
    for _ in tqdm(range(nb_epochs)):
        for batch in data_loader:
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)

            regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            # print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()
        torch.save(nerf_model.cpu(), 'nerf_model')
        nerf_model.to(device)
        torch.cuda.empty_cache()

    return training_loss


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_dataset = torch.from_numpy(np.load('rtrain.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('rtrain.pkl', allow_pickle=True))
    
    model = FastNerf().to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    data_loader = DataLoader(training_dataset, batch_size=512, shuffle=True)  # Reduced batch size
    train(model, model_optimizer, scheduler, data_loader, device=device, hn=2, hf=6, nb_epochs=16, nb_bins=192)
    #torch.cuda.empty_cache()
    model
    #cache = Cache(model, 2.2, device, 192, 128)
    # for idx in range(200):
    #     test(cache, 2., 6., testing_dataset, img_index=idx, nb_bins=192, H=400, W=400)


