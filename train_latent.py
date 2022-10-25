# Train latent diffusion



import os
from tqdm.auto import tqdm
from opt import config_parser
import logging
import ruamel.yaml 
yaml2 = ruamel.yaml.YAML()
from utils import set_logger, printlog
from collections import OrderedDict


import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
import datetime
import skimage

from dataLoader import dataset_dict
import sys
import pdb
import time


# load dreamfusion
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class MainApp():
    def __init__(self, args):
        self.args = args
        self.init_dreamfusion()
        self.init_tensorf()

    def render(self):
        args = self.args
        image_dir = os.path.join(self.logfolder, 'imgs_path_all')
        os.makedirs(image_dir, exist_ok=True)
        dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=8)
        with torch.no_grad():
            print("rendering image...")
            for batch_id, batch in enumerate(tqdm(dataloader)):
                rays_train = batch['rays'].squeeze().to(self.device)               
                ret = renderer(
                    rays_train, 
                    self.tensorf, 
                    chunk=args.batch_size,
                    N_samples=self.nSamples,
                    white_bg = False,
                    ndc_ray=args.ndc_ray,
                    device=device,
                    is_train=False
                )
                latents = ret[0].view(64,64,4).permute(2,0,1)[None]
                
                image = self.decode_latents(latents)
                image = image.cpu()[0].permute(1,2,0).numpy()
                image = skimage.img_as_ubyte(image)
                skimage.io.imsave(os.path.join(image_dir, f"{batch_id:03d}.png"), image)
                
                depth_map = ret[2].view(64,64).cpu().numpy()
                depth_map, _ = visualize_depth_numpy(depth_map,self.test_dataset.near_far)
                depth_map = skimage.img_as_ubyte(depth_map)
                skimage.io.imsave(os.path.join(image_dir, f"{batch_id:03d}_depth.png"), depth_map)



    def init_tensorf(self):
        args = self.args
        # init log file
        if args.add_timestamp:
            logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
        else:
            logfolder = f'{args.basedir}/{args.expname}'
        
        self.logfolder = logfolder
        os.makedirs(logfolder, exist_ok=True)
        os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
        os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
        os.makedirs(f'{logfolder}/rgba', exist_ok=True)
        gfile_stream = open(os.path.join(logfolder, 'stdout.txt'), 'w')
        set_logger(gfile_stream)
        

        prompt = "a high quality photo of a pineapple"
        dataset = dataset_dict[args.dataset_name]
        self.train_dataset = dataset(split="train", iter=args.n_iters)
        self.test_dataset = dataset(prompt, split="test", iter=120)

        aabb = self.train_dataset.scene_bbox.to(device)
        near_far =  self.train_dataset.near_far
        reso_cur = N_to_reso(args.N_voxel_init, aabb)
        self.nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

        if args.ckpt is not None:
            ckpt = torch.load(args.ckpt, map_location=device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device':device})
            tensorf = eval(args.model_name)(**kwargs)
            tensorf.load(ckpt)
        else:
            # Pure: Dynamic Ordered dict for easily design a model without conflict 
            kwargs = OrderedDict([
                ("aabb", aabb),
                ("gridSize", reso_cur),
                ("device", device),
                ("density_n_comp", args.n_lamb_sigma),
                ("appearance_n_comp", args.n_lamb_sh),
                ("app_dim", args.data_dim_color),
                ("near_far", near_far),
                ("shadingMode", args.shadingMode),
                ("alphaMask_thres", args.alpha_mask_thre),
                ("density_shift", args.density_shift),
                ("distance_scale", args.distance_scale),
                ("pos_pe",args.pos_pe),
                ("view_pe",args.view_pe), 
                ("fea_pe", args.fea_pe),
                ("featureC", args.featureC), 
                ("step_ratio", args.step_ratio), 
                ("fea2denseAct", args.fea2denseAct)
            ])    
            if args.model_name in ['Sigma5DRF']:
                kwargs['dataset']=  self.train_dataset
            if args.num_frames > 1: #only some model support max_t, so we pass max_t if num_frames provide
                kwargs["max_t"] = args.num_frames 
                kwargs["t_keyframe"] = args.t_keyframe
                kwargs["upsamp_list"] = args.upsamp_list
            tensorf = eval(args.model_name)(**kwargs)

        self.tensorf = tensorf
        grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
        # assign learning rate
        if args.lr_decay_iters > 0:
            self.lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
        else:
            self.args.lr_decay_iters = args.n_iters
            self.lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        self.summary_writer = SummaryWriter(self.logfolder)
        

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


    def update_learning_rate(self, step_id = 0):
        for group_id, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * self.lr_factor
            self.summary_writer.add_scalar(f'learning_rate/{group_id:02d}', param_group['lr'], global_step=step_id)


    def reconstruction(self):
        args = self.args
        # training loop
        with tqdm(total=args.n_iters, file=sys.stdout) as pbar:
            #step_id = 0 #NOTE resume training is not support yet!
            dataloader = DataLoader(self.train_dataset, batch_size=1, num_workers=8)
            for step_id, batch in enumerate(dataloader):
                self.optimizer.zero_grad()
                rays_train = batch['rays'].squeeze().to(self.device) #[4096, 6]
                text_embed = batch['text_embed'].squeeze().to(self.device) #[2, 77, 768]
               
                ret = renderer(
                    rays_train, 
                    self.tensorf, 
                    chunk=args.batch_size,
                    N_samples=self.nSamples,
                    white_bg = False,
                    ndc_ray=args.ndc_ray,
                    device=device,
                    is_train=True
                )
                latents = ret[0].view(64,64,4).permute(2,0,1)[None]

                
                self.dreamfusion_step(text_embed, latents) #update gradient (backward)
                self.optimizer.step()
                self.update_learning_rate(step_id)

                # tensorboard log
                latent_min = torch.min(latents).detach().item()
                latent_max = torch.max(latents).detach().item()
                latent_diff = latent_max - latent_min
                self.summary_writer.add_scalar('latents/min', latent_min, global_step=step_id)
                self.summary_writer.add_scalar('latents/max', latent_max, global_step=step_id)
                self.summary_writer.add_scalar('latents/distance', latent_diff, global_step=step_id)
                if step_id % 10 == 0:
                    self.validate_image(0, step_id)
                    self.validate_image(30, step_id)
                    self.validate_image(60, step_id)
                    self.validate_image(90, step_id)

                if args.checkpoint_every > 0 and step_id % args.checkpoint_every == 0 and step_id > 0:
                    self.tensorf.save(f'{self.logfolder}/{args.expname}_{step_id:06d}.th')
                


                # update progress bar
                pbar.update(1)

        # save checkpoint before finished        
        self.tensorf.save(f'{self.logfolder}/{args.expname}.th')

    def validate_image(self, img_id, step_id):
        batch = self.test_dataset[img_id]
        rays_train = batch['rays'].squeeze().to(self.device)               
        ret = renderer(
            rays_train, 
            self.tensorf, 
            chunk=args.batch_size,
            N_samples=self.nSamples,
            white_bg = False,
            ndc_ray=args.ndc_ray,
            device=device,
            is_train=False
        )
        latents = ret[0].view(64,64,4).permute(2,0,1)[None]        
        image = self.decode_latents(latents)
        depth_map = ret[2].view(64,64).cpu().numpy()
        latent_grid = torchvision.utils.make_grid(torch.sigmoid(latents)[0][:,None])
        depth_map, _ = visualize_depth_numpy(depth_map,self.test_dataset.near_far)
        depth_map = torch.from_numpy(depth_map).permute(2,0,1)

        self.summary_writer.add_image(f'validation_{img_id}/predict', image[0], step_id)
        self.summary_writer.add_image(f'validation_{img_id}/latent_sigmoid', latent_grid, step_id)
        self.summary_writer.add_image(f'validation_{img_id}/depth', depth_map, step_id)



    def init_dreamfusion(self):
        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=self.token).to(self.device)
        
        # 4. Create a scheduler for inference
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')


    def dreamfusion_step(self, text_embeddings, latents, guidance_scale=100):
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        latents.backward(gradient=grad, retain_graph=True)


if __name__ == '__main__':
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20121202)
    np.random.seed(20121202)

    args = config_parser()
    print(args)

    app = MainApp(args)
    if args.render_only: #and (args.render_test or args.render_path):
        app.render()
    else:
        app.reconstruction()

