
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

from dataLoader import dataset_dict
import sys
import pdb
import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


#@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, ndc_ray=args.ndc_ray, max_t=args.num_frames, hold_every=args.hold_every)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    if args.num_frames > 1: #only some model support max_t, so we pass max_t if num_frames provide
        kwargs.update({'max_t': args.num_frames})
        kwargs.update({'t_keyframe': args.t_keyframe})
        kwargs.update({'upsamp_list': args.upsamp_list})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    #pdb.set_trace()
    if args.model_name in ['TensorSph']:
        tensorf.set_origin(test_dataset.origin,test_dataset.sph_box,test_dataset.sph_frontback)
    all_rays = test_dataset.all_rays
    this_rays = all_rays[0,:100]
    tensorf_for_renderer = tensorf 
    if args.data_parallel:
        tensorf_for_renderer = torch.nn.DataParallel(tensorf)
    #rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(this_rays, tensorf_for_renderer, chunk=args.batch_size,  N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

    logfolder = os.path.dirname(args.ckpt)
    
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, ndc_ray=args.ndc_ray, max_t=args.num_frames, hold_every=args.hold_every)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False, ndc_ray=args.ndc_ray, max_t=args.num_frames, hold_every=args.hold_every)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True, ndc_ray=args.ndc_ray, max_t=args.num_frames, hold_every=args.hold_every)
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
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        """
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    grid_level=args.grid_level, grid_feature_per_level=args.grid_feature_per_level, grid_hash_log2=args.grid_hash_log2, grid_base_resolution=args.grid_base_resolution, grid_level_scale=args.grid_level_scale,
                )
        """
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
        if args.num_frames > 1: #only some model support max_t, so we pass max_t if num_frames provide
            kwargs["max_t"] = args.num_frames 
            kwargs["t_keyframe"] = args.t_keyframe
            kwargs["upsamp_list"] = args.upsamp_list
        tensorf = eval(args.model_name)(**kwargs)
    if args.model_name in ['TensorSph']:
        tensorf.set_origin(train_dataset.origin,train_dataset.sph_box,train_dataset.sph_frontback)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    printlog(f"lr decay {args.lr_decay_target_ratio} {args.lr_decay_iters}")
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]      

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray: 
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    printlog(f"initial Ortho_reg_weight {Ortho_reg_weight}")

    L1_reg_weight = args.L1_weight_inital
    printlog(f"initial L1_reg_weight {L1_reg_weight}")
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    printlog(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    scaler = GradScaler()
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    tensorf_for_renderer = tensorf 
    if args.data_parallel:
        tensorf_for_renderer = torch.nn.DataParallel(tensorf)
    start_time = time.time()
    for iteration in pbar:
        with autocast(enabled=False):
            #if iteration %50 > 5 and iteration %50 < 48 : continue

            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

            #rgb_map, alphas_map, depth_map, weights, uncertainty
            rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf_for_renderer, chunk=args.batch_size,
                                    N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

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
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.zero_grad()
            #try:
            total_loss.backward()
            #except:
            #    print("Something wrong with backward()")
            #    pdb.set_trace()
            optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)
        summary_writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step=iteration)
        
        if False and iteration % (args.n_iters // 20) == 0:
            with torch.no_grad():
                print("rendering images... at step {}".format(iteration))
                W, H = test_dataset.img_wh
                samples = test_dataset.all_rays[0]
                rays = samples.view(-1,samples.shape[-1])
                rgb_map, _, depth_map, _, _ = renderer(rays, tensorf_for_renderer, chunk=256, N_samples=nSamples, ndc_ray=ndc_ray, white_bg = white_bg, device=device)
                rgb_map = rgb_map.clamp(0.0, 1.0)
                rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
                summary_writer.add_image('train/eval_image', rgb_map.permute(2,0,1), global_step=iteration)
                near_far = test_dataset.near_far
                depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)            
                summary_writer.add_image('train/eval_depth', torch.from_numpy(depth_map).permute(2,0,1), global_step=iteration)

        
        if False and args.visualize_tensor > 0 and ((iteration % 101 == 0 and iteration<5000) or (iteration % 1001 == 0 and iteration>5000)):
            torch.cuda.empty_cache()
            #pdb.set_trace()
            W, H = test_dataset.img_wh
            with torch.no_grad():
                rays = test_dataset.all_rays[0]
                #pdb.set_trace()
                rgb_map, _, _, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=ndc_ray, white_bg = white_bg, device=device)
                rgb_map = rgb_map.reshape(H, W, 3)

                rgb_map = (rgb_map.cpu().detach().numpy() * 255).astype('uint8')
                imageio.imwrite(f'script/snap.png', rgb_map)
                tensor = (tensorf.density_plane[0].cpu().detach().numpy() * 255).astype('uint8')
                imageio.imwrite(f'script/tensor.png', tensor[0,:3].transpose(1,2,0))

            #pdb.set_trace()
            
        

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
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            logging.info(f'Iteration {iteration} test psnr {np.mean(PSNRs_test)}')



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
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


        if iteration in upsamp_list:
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
        
        if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0 and iteration > 0:
            tensorf.save(f'{logfolder}/{args.expname}_{iteration:06d}.th')
        

    tensorf.save(f'{logfolder}/{args.expname}.th')
    print("Trained finished in {} seconds".format(time.time() - start_time))

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True, ndc_ray=args.ndc_ray, max_t=args.num_frames, hold_every=args.hold_every)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        printlog(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


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

