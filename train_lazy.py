# Training script with LazyLoader 
#
# Instead of dumping all input into memory, we lazy load on the fly.
# This can create an IO bound where slow training down but helping to training large dataset such as MetaVideoLazy



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
import datetime
from torch.utils.data import DataLoader

from dataLoader import dataset_dict
import sys
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


@torch.no_grad()
def evaluation_lazy(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    #os.makedirs(savePath+'/img', exist_ok=True)
    os.makedirs(savePath+"/img/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else test_dataset.all_rays.shape[0] // N_vis

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(os.cpu_count() * args.dataloader_thread_ratio))

    for idx, samples in tqdm(enumerate(test_dataloader), file=sys.stdout):
        if N_vis > 0 and idx % N_vis != 0: continue

        W, H = test_dataset.img_wh
        rays = samples['rays'].view(-1,samples['rays'].shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=512, N_samples=N_samples, ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, min_max = visualize_depth_numpy(depth_map.numpy(),near_far)
        if True: #temporary predict
            gt_rgb = samples['rgbs'].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/img/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/img/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path_lazy(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/img/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    for idx, c2w in enumerate(tqdm(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        if hasattr(test_dataset, 'max_t'):
            rays = torch.cat([rays_o, rays_d, torch.ones_like(rays_o[:, :1]) * idx], 1)
        else: 
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
 

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=512, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/img/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/img/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = get_dataset(args, 'test')
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    if args.num_frames > 1 or args.model_name == 'TensoRFVideo': #only some model support max_t, so we pass max_t if num_frames provide
        kwargs.update({'max_t': args.num_frames})
        kwargs.update({'t_keyframe': args.t_keyframe})
        kwargs.update({'upsamp_list': args.upsamp_list})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    #pdb.set_trace()
    if args.model_name in ['TensorSph']:
        tensorf.set_origin(test_dataset.origin,test_dataset.sph_box,test_dataset.sph_frontback)
    tensorf_for_renderer = tensorf 
    if args.data_parallel:
        tensorf_for_renderer = torch.nn.DataParallel(tensorf)

    logfolder = os.path.dirname(args.ckpt)

    

    if False and args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = get_dataset(args, 'train')
        train_dataset.is_sampling = False
        PSNRs_test = evaluation_lazy(train_dataset,tensorf_for_renderer, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


    if True or args.render_test:
        test_dataset = get_dataset(args, 'test')
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_lazy(test_dataset,tensorf_for_renderer, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if False and args.render_dynerf:
        test_dataset = get_dataset(args, 'test', hold_every_frame=10)
        os.makedirs(f'{logfolder}/imgs_test_dynerf', exist_ok=True)
        PSNRs_test = evaluation_lazy(test_dataset,tensorf_for_renderer, args, renderer, f'{logfolder}/imgs_test_dynerf/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test_dynerf psnr: {np.mean(PSNRs_test)} <========================')

    if True or args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path_lazy(test_dataset,tensorf_for_renderer, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


def get_dataset(args, split, hold_every_frame=1, psudo_length=-1):
    dataset_class = dataset_dict[args.dataset_name]
    dataset = dataset_class(
        args.datadir,
        split=split,
        downsample=args.downsample_train,
        is_stack=(split == False),
        ndc_ray=args.ndc_ray,
        max_t=args.num_frames,
        hold_every=args.hold_every,
        num_rays=args.batch_size,
        hold_every_frame=hold_every_frame,
        psudo_length=psudo_length
    )
    return dataset

def reconstruction(args):

    train_dataset = get_dataset(args, 'train')
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    gfile_stream = open(os.path.join(logfolder, 'stdout.txt'), 'w')
    set_logger(gfile_stream)
    printlog('Start Training')
    summary_writer = SummaryWriter(logfolder)
    with open(os.path.join(logfolder, "config.yml"), "w") as f:
        yaml2.dump(vars(args), f)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)

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
            ("density_n_comp", n_lamb_sigma),
            ("appearance_n_comp", n_lamb_sh),
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
        if args.num_frames > 1 or args.model_name == 'TensoRFVideo': #only some model support max_t, so we pass max_t if num_frames provide
            kwargs["max_t"] = args.num_frames 
            kwargs["t_keyframe"] = args.t_keyframe
            kwargs["upsamp_list"] = args.upsamp_list
        tensorf = eval(args.model_name)(**kwargs)
    if args.model_name in ['TensorSph']:
        tensorf.set_origin(train_dataset.origin,train_dataset.sph_box,train_dataset.sph_frontback)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    scaler = GradScaler()
    
    training_loop(tensorf, optimizer, scaler, summary_writer, args=args, hierarchy_type='coarse') #key frame training
    training_loop(tensorf, optimizer, scaler, summary_writer, args=args, hierarchy_type='fine') #all frame trainign

    tensorf.save(f'{logfolder}/{args.expname}.th')
    
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = get_dataset(args, 'train')
        train_dataset.is_sampling = False
        PSNRs_test = evaluation_lazy(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        test_dataset = get_dataset(args, 'test')
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_lazy(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)

    if args.render_dynerf:
        test_dataset = get_dataset(args, 'test', hold_every_frame=10)
        os.makedirs(f'{logfolder}/imgs_test_dynerf', exist_ok=True)
        PSNRs_test = evaluation_lazy(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_dynerf/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test_dynerf psnr: {np.mean(PSNRs_test)} <========================')
        summary_writer.add_scalar('test_dynerf/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)
    
    if args.render_firstframe:
        test_dataset = get_dataset(args, 'test', hold_every_frame=1000000)
        os.makedirs(f'{logfolder}/imgs_test_dynerf', exist_ok=True)
        PSNRs_test = evaluation_lazy(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_firstframe/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test_firstframe psnr: {np.mean(PSNRs_test)} <========================')
        summary_writer.add_scalar('test_dynerf/psnr_all', np.mean(PSNRs_test), global_step=args.n_iters)

    if args.render_path:
        raise NotImplementError()
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def training_loop(tensorf, optimizer, scaler, summary_writer, args, hierarchy_type='coarse') :
    n_iters = args.keyframe_iters if hierarchy_type=='coarse' else args.n_iters

    hold_every_frame = 1# args.t_keyframe if hierarchy_type=='coarse' else 1
    train_dataset = get_dataset(args, 'train', hold_every_frame= hold_every_frame, psudo_length=n_iters)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=int(os.cpu_count() * args.dataloader_thread_ratio))


    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    if hierarchy_type=='coarse' or args.keyframe_iters < 0:
        if args.lr_decay_iters > 0:
            lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
            lr_decay_iters = args.lr_decay_iters
        else:
            lr_decay_iters = n_iters
            lr_factor = args.lr_decay_target_ratio**(1/n_iters)
        printlog(f"lr decay {args.lr_decay_target_ratio} {lr_decay_iters}")
    else:
        printlog(f"continue tuning without decay")
        # continue training without further more deacy in fine step
        lr_factor = 1.0 
        TV_weight_density *= args.lr_decay_target_ratio
        TV_weight_app *= args.lr_decay_target_ratio

    reso_mask = None

    #linear in logrithmic space, note that we can upsampling only coarse
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]      
    ndc_ray = args.ndc_ray
    
    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    
    if not args.ndc_ray: 
        raise NotImplementError('haven\'t implement filter ray to support non-ndc mode yet')
        allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)

    Ortho_reg_weight = args.Ortho_weight
    L1_reg_weight = args.L1_weight_inital
    
    tvreg = TVLoss()
 
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init if (hierarchy_type=='coarse' or args.keyframe_iters < 0) else args.N_voxel_final, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    if hierarchy_type == 'coarse':
        print("==== Training Coarse (keyframe) level ====")
        printlog(f"initial Ortho_reg_weight {Ortho_reg_weight}")
        printlog(f"initial L1_reg_weight {L1_reg_weight}")
        printlog(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    else:
        print("==== Training Fine (all-frame) level ====")



    pbar = tqdm(range(n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    tensorf_for_renderer = tensorf 
    if args.data_parallel:
        tensorf_for_renderer = torch.nn.DataParallel(tensorf)


    median_step = int(args.median_ratio * n_iters)
    temporal_step = int(args.temporal_ratio * n_iters)
    train_iterator = iter(train_dataloader)
    if hierarchy_type == 'coarse' and args.median_keyframe:
        train_dataloader.dataset.is_median = True 
    with autocast(enabled=False):
        for iteration in pbar:
            #enable weight sampling option
            if hierarchy_type == 'fine':
                if iteration == median_step:
                    print("apply median sampling...")
                    train_dataloader.dataset.is_median = True 
                    train_dataloader.dataset.is_temporal = False 
                if iteration == temporal_step:
                    print("apply temporal sampling...")                
                    train_dataloader.dataset.is_median = False 
                    train_dataloader.dataset.is_temporal = True
            # pick ray_batch from traintring loader
            try:
                ray_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                ray_batch = next(train_iterator)
            rays_train = ray_batch['rays'][0]
            rgb_train =  ray_batch['rgbs'][0].to(device)

            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf_for_renderer, chunk=args.batch_size, N_samples=nSamples, white_bg = train_dataset.white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
            
            loss = torch.mean((rgb_map - rgb_train) ** 2)

            # loss
            total_loss = loss
            if iteration % args.TV_every==0:
                if Ortho_reg_weight > 0:
                    loss_reg = tensorf.vector_comp_diffs()
                    total_loss += Ortho_reg_weight*loss_reg
                    summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
                if L1_reg_weight > 0:
                    loss_reg_L1 = tensorf.density_L1()
                    total_loss += L1_reg_weight*loss_reg_L1
                    summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

                if TV_weight_density>0:
                    TV_weight_density *= lr_factor
                    loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
                if TV_weight_app>0:
                    TV_weight_app *= lr_factor
                    loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app
                    total_loss = total_loss + loss_tv
                    summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
            if args.grad_scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            loss = loss.detach().item()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
            summary_writer.add_scalar('train/mse', loss, global_step=iteration)
            summary_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=iteration)

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                    + f' mse = {loss:.6f}'
                )
                PSNRs = []
                


            if iteration % args.vis_every == args.vis_every - 1:
                PSNRs_test = evaluation_lazy(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                        prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
                summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
                logging.info(f'Iteration {iteration} test psnr {np.mean(PSNRs_test)}')



            if iteration in update_AlphaMask_list:

                if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                    reso_mask = reso_cur
                if reso_mask == None:
                    reso_mask = tuple([256,256,256])
                new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
                if iteration == update_AlphaMask_list[0]:
                    tensorf.shrink(new_aabb)
                    # tensorVM.alphaMask = None
                    L1_reg_weight = args.L1_weight_rest
                    printlog(f"continuing L1_reg_weight {L1_reg_weight}")


                if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                    # filter rays outside the bbox
                    allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


            # currently, upsammling uspo
            if (hierarchy_type == 'coarse' or args.keyframe_iters < 0) and iteration in upsamp_list:
                n_voxels = N_voxel_list.pop(0)
                reso_cur = N_to_reso(n_voxels, tensorf.aabb)
                nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
                print("Resolution ====== > ")
                print(reso_cur)
                tensorf.upsample_volume_grid(reso_cur)
                torch.cuda.empty_cache()

                if args.lr_upsample_reset:
                    printlog("reset lr to initial")
                    lr_scale = 1 #0.1 ** (iteration / args.n_iters)
                else:
                    lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))


if __name__ == '__main__':
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20121202)
    np.random.seed(20121202)

    args = config_parser()
    print(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)
