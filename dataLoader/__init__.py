from dataLoader.llff_fern400 import LLFFFern400
from .llff import LLFFDataset
from .shiny import ShinyDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .deepvv import DeepvvDataset
from .meta import MetaVideoDataset
from .meta_lazy import MetaVideoLazyDataset
from .meta_dynamic import MetaDynamicDataset
from .meta_crop import MetaCropVideoDataset
from .llff_cube import LLFFCube
from .shiny_few import ShinyFew1, ShinyFew5, ShinyFew10, ShinyFew15, ShinyFew20, ShinyFern400
from .llff_fern400 import LLFFFern400

dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
               'shiny':ShinyDataset,
               'deepvv':DeepvvDataset,
               'meta':MetaVideoDataset,
               'meta_lazy': MetaVideoLazyDataset,
               'meta_dynamic': MetaDynamicDataset,
               'meta_crop': MetaCropVideoDataset,
               'llff_cube': LLFFCube,
               'shiny_few1': ShinyFew1,
               'shiny_few5': ShinyFew5,
               'shiny_few10': ShinyFew10,
               'shiny_few15': ShinyFew15,
               'shiny_few20': ShinyFew20,
               'shiny_fern400': ShinyFern400,
               'llff_fern400': LLFFFern400
               }