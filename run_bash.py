import os

batch_size = 4096
N_vis = 5

n_iters = 25000
for data_name in ['labjuice','optical_bottle','wrench']: # 
    cmd = f'python train.py --config configs2/defult.txt ' \
          f'--dataset_name shiny --datadir data/shiny_for_eval/{data_name} --downsample_train 2.0 --ndc_ray 1 '\
          f'--expname {data_name}  --basedir log/0327/baseline2/  --add_timestamp 0   ' \
          f'--batch_size {batch_size}  --n_iters {n_iters}  ' \
          f'--N_voxel_init {128**3} --N_voxel_final {750**3} '\
          f'--N_vis {N_vis}  ' \
          f'--shadingMode MLP_Fea --fea2denseAct relu --view_pe {0} --fea_pe {0} ' \
          f'--TV_weight_density {1.0} --TV_weight_app {1.0} ' \
          f'--render_test 1 --render_path 1  ' #\
         # f'--model_name  TensorF5D  '

    print(cmd)
    os.system(cmd)