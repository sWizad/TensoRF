import torch
import numpy as np
from .llff import LLFFDataset

class LLFFCube(LLFFDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recalcurate_bound()

    def recalcurate_bound(self):
        self.scene_bbox = torch.tensor([[-1.0, -1.0,  -1.0], [1.0, 1.0, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    """
    def recalcurate_bound(self):
        print("recalcurate bound...")
        poses = self.poses
        bottom_line = np.zeros_like(self.poses[:,:1,:])
        bottom_line[:,:,3] = 1.0
        poses = np.concatenate([self.poses, bottom_line],axis=1) #20,
        homo_nearfar = self.near_fars[..., None] #20,2,1
        nearfar_line = np.zeros_like(homo_nearfar) #20,2,1
        homo_nearfar = np.concatenate([nearfar_line, nearfar_line, homo_nearfar, np.ones_like(nearfar_line)], axis=-1)[...,None] #20,2,4,1
        points = np.concatenate([poses[:,None],poses[:,None]], axis=1) @ homo_nearfar
        points = points.reshape((-1,4))
        
        for i in range(3):
            points[...,i] = points[...,i] / points[...,3] 
        points = points[...,:3]

        self.scene_bbox = torch.from_numpy(
            np.concatenate([
                np.min(points,axis=0).reshape(1,3),
                np.max(points,axis=0).reshape(1,3)
            ], axis=0)
        )
        print("SCENE_BBOX: ", self.scene_bbox)    
        #self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.scene_bbox = torch.tensor([[-1.2023, -0.3734,  -1.2699], [1.4207,  0.4175,  6.2730]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
    """