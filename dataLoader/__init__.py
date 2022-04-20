from .llff import LLFFDataset
from .shiny import ShinyDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .deepvv import DeepvvDataset
from .meta import MetaVideoDataset
from .meta_lazy import MetaVideoLazyDataset
from .meta_dynamic import MetaDynamicDataset


dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
               'shiny':ShinyDataset,
               'deepvv':DeepvvDataset,
               'meta':MetaVideoDataset,
               'meta_lazy': MetaVideoLazyDataset,
               'meta_dynamic': MetaDynamicDataset}