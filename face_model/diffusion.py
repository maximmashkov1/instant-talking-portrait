import torch
from torch import nn
import math
import torch.nn.functional as F

    
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) 

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        time_emb = self.mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1) 
        scale_shift = time_emb.chunk(2, dim=1) #returns one value

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(b, self.heads, -1, n)
        k = k.view(b, self.heads, -1, n)
        v = v.view(b, self.heads, -1, n)
        q = q * self.scale
        sim = torch.matmul(q.permute(0, 1, 3, 2), k)
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v.permute(0, 1, 3, 2))
        out = out.permute(0, 1, 3, 2).reshape(b, -1, n)
        return self.to_out(out) + x

class Unet1d(nn.Module):
    def __init__(self, normalization=None, channels=[4, 8, 16, 32], fourier_dim=8, in_cond_dim=25, gru_layers=1, gru_dim=25, cond_dim=8, timesteps=100):
        super().__init__()
        
        self.gru = nn.GRU(input_size=in_cond_dim, hidden_size=gru_dim, num_layers=gru_layers, batch_first=True, bidirectional=False, dropout=0)
        self.gru_dim = gru_dim
        self.cond_mapping = nn.Sequential(nn.Linear(gru_dim, cond_dim), nn.LeakyReLU())
        self.conditioning=None
        
        self.normalization = normalization
        self.device=None
        self.cond_dim = cond_dim         
        self.num_timesteps = timesteps
        time_dim = fourier_dim * 4
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.downs = nn.ModuleList([])
        
        channels[0] = 2+cond_dim
        
        for i in range(len(channels) - 1):
            self.downs.append(nn.ModuleList([ResnetBlock(channels[i], channels[i+1], time_dim),
                              ResnetBlock(channels[i+1], channels[i+1], time_dim)]))

        mid_dim = channels[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)
        
        self.ups = nn.ModuleList([])
        
        for i in reversed(range(len(channels) - 1)):
            self.ups.append(nn.ModuleList([ResnetBlock(channels[i+1] * 2, channels[i], time_dim),
                           ResnetBlock(channels[i], channels[i], time_dim)]))
        
        self.out_conv = nn.Conv1d(channels[0], 2, 1, padding=0)
        
        self.betas = linear_beta_schedule(timesteps).to(torch.float32)
        self.alphas = 1. - self.betas
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.set_device('cuda')
        
    def set_device(self,device):
        self.device=device
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.nn.Module) or isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
        self.to(device)

    def compute_cond(self, conditioning, size):
        gru_output = self.gru(conditioning)[0]
        shape_initial = gru_output.clone().shape
        gru_output_mapped = self.cond_mapping(gru_output.reshape(-1, self.gru_dim))
        gru_output_reshaped = gru_output_mapped.reshape(shape_initial[0], shape_initial[1], self.cond_dim)
        #cond_compressed = F.interpolate(gru_output_reshaped.transpose(1, 2), size=size, mode='linear', align_corners=False)
        self.conditioning = gru_output_reshaped.transpose(1, 2)
        

    def forward(self, x, time, conditioning):
        
            
        t = self.time_mlp(time)

        if self.conditioning == None:
            self.compute_cond(conditioning, x.shape[-1])
                              
        x=torch.cat((x, self.conditioning), dim=1)

        residuals = []
        print(x.shape)
        for (down1, down2) in self.downs:
            x = down1(x, t)
            x = down2(x, t)
            residuals.append(x)
            
            if x.shape[-1] % 2 != 0:
                x = F.pad(x, (0, 1))
                
            x = F.avg_pool1d(x, 2)
            
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)   
        
        for (up1, up2), res in zip(self.ups, reversed(residuals)):
            x = F.interpolate(x, scale_factor=2, mode='linear')            
            if x.shape[-1] > res.shape[-1]:
                x = x[:, :, :res.shape[-1]]
            elif x.shape[-1] < res.shape[-1]:
                res = res[:, :, :x.shape[-1]]
            x = torch.cat([x, res], dim=1)
            x = up1(x, t)
            x = up2(x, t)
            
        x = self.out_conv(x)
        return x
        
    @torch.no_grad()
    def ddim_sample_timestep(self, x, t, conditioning, eta=0.0):
        alphas_t = extract(self.alphas_cumprod, t, x.shape)
        alphas_t_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sqrt_one_minus_alphas_t = torch.sqrt(1. - alphas_t)
        
        noise_modeled = self.forward(x, t, conditioning)
        pred_x0 = (x - sqrt_one_minus_alphas_t * noise_modeled) / torch.sqrt(alphas_t)
        
        if t == 0:
            return pred_x0
        else:
            sigma = eta * torch.sqrt((1 - alphas_t_prev) / (1 - alphas_t)) * torch.sqrt(1 - alphas_t / alphas_t_prev)
            noise = torch.randn_like(x)
            return torch.sqrt(alphas_t_prev) * pred_x0 + torch.sqrt(1 - alphas_t_prev - sigma**2) * noise_modeled + sigma * noise
            
    @torch.no_grad()
    def generate_ddim(self, conditioning, timesteps=None, eta=0.0):
        self.eval()
        self.conditioning=None
        if len(conditioning.shape) < 2:
            conditioning = conditioning.unsqueeze(0)
        x = torch.randn(1, 2, conditioning.shape[1]).to(self.device)
        
        if timesteps is None:
            timesteps = self.num_timesteps
        times = torch.linspace(0, timesteps - 1, steps=timesteps)
        times = torch.tensor(list(reversed(times.int().tolist())), dtype=torch.long).to(self.device)

        for t in times:
            x = self.ddim_sample_timestep(x, t.unsqueeze(0), conditioning, eta)
        self.conditioning=None
        return x
  
    def get_loss(self, x_0, conditioning, t):
        
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        pred_noise = self.forward(x_t, t, conditioning) 
        self.conditioning=None
        #alphas_t = extract(self.alphas_cumprod, t, x_0.shape)
        #sqrt_one_minus_alphas_t = torch.sqrt(1. - alphas_t)
        #pred_x0 = (x_t - sqrt_one_minus_alphas_t * pred_noise) / torch.sqrt(alphas_t)

        #norm_loss = torch.pow(pred_x0, 2).mean()
        noise_loss = F.mse_loss(pred_noise, noise, reduction='mean')
        #print(norm_loss.item(), noise_loss.item())
        loss = noise_loss#+norm_loss
        
        return loss
    @torch.no_grad()
    def ddpm_sample_timestep(self, x, t, conditioning):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        noise_modeled = self.forward(x, t, conditioning)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_modeled / sqrt_one_minus_alphas_cumprod_t)
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
            
    @torch.no_grad()        
    def generate_ddpm(self, conditioning):
        self.eval()
        self.conditioning=None
        if len(conditioning.shape) < 2:
            conditioning=conditioning.unsqueeze(0)
        x = torch.randn(1,2,conditioning.shape[1]).to(self.device)
        
        times = torch.linspace(0, self.num_timesteps - 1, steps=self.num_timesteps)  
        times = torch.tensor(list(reversed(times.int().tolist())), dtype=torch.long).to(self.device)

        for t in times:
            x = self.ddpm_sample_timestep(x, t.unsqueeze(0), conditioning)

        self.conditioning=None
        return x
  

def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps, dtype = torch.float64)

