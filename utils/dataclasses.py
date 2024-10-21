from .networks import Mozafari2018, STDPNet
from dataclasses import dataclass
    
@dataclass
class ExperimentInfoGlobal:
    use_cuda:bool = True
    log_dir:str = "log"

global expr_info
expr_info_global = ExperimentInfoGlobal()

@dataclass
class ExperimentInfoLocal:
    log_name:str = "log"
    num_steps:int = 15
    net_type:type[STDPNet] = Mozafari2018
    seed:int = 42
    delta:int = 1
    data_root:str = "data/"