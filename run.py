import easydict
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from src.pretreatment.dataload import StoreDataset, DataloaderType
from src.train.train import train

if __name__ == '__main__':
    config_path = "./src/config.yml"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # loading config file
    config = EasyDict(yaml.load(open(config_path, "r"), Loader=yaml.FullLoader))
    # loading datasets
    train_dataset = StoreDataset(config=config, data_type=DataloaderType.train)
    train_dataset_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=False)

    val_dataset = StoreDataset(config=config, data_type=DataloaderType.validate)
    val_dataset_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
    # test_dataset = SensorDataset(config=config,
    #                              is_train=False,
    #                              training_length=config.train.training_length,
    #                              forecast_window=config.train.forecast_window)
    # test_dataset_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = train(config=config,
                  dataloader=train_dataset_loader,
                  val_dataloader=val_dataset_loader,
                  path_to_save_model=config.save.model_path,
                  path_to_save_loss=config.save.loss_path,
                  path_to_save_predictions=config.save.predictions_path,
                  device=device)


