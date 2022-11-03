import easydict
import yaml
from easydict import EasyDict

from src.pretreatment.dataload import load

if __name__ == '__main__':
    config_path = "./src/config.yml"
    # loading config file
    config = EasyDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    load(config)
