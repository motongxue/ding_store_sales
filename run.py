import easydict
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from src.pretreatment.dataload import data_load
from src.train.train import transformer

if __name__ == '__main__':
    config_path = "./src/config.yml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loading config file
    config = EasyDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))

    train_loader, test_loader = data_load(config)

    # ?
    epoch = 1000
    k = 60
    model = transformer(dataloader=train_loader,
                        EPOCH=epoch,
                        k=k,
                        path_to_save_model=config.save.model_path,
                        path_to_save_loss=config.save.loss_path,
                        path_to_save_predictions=config.save.predictions_path,
                        device=device)


