from scripts.network import VAE
from scripts.data_loader import CarlaAEBSDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import transforms
from scipy import stats
import os

class ICAD():
    def __init__(self, data_path):

        self.mse = torch.nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = VAE()
        self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load("./model/vae.pt", map_location=self.device))
        self.net.eval()
        self.nc_calibration = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        try:
            self.nc_calibration = np.load("./data/icad/calibration.npy")
        except:
            dataset = CarlaAEBSDataset(data_path, "calibration")
            data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
            for idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    x, mu, logvar = self.net(inputs)
                    nc = self.mse(inputs, x)
                    self.nc_calibration.append(nc)

            self.nc_calibration = np.asarray(self.nc_calibration)
            if not os.path.exists("./data/calibration"):
                os.makedirs("./data/calibration")
            np.save("./data/icad/calibration.npy", self.nc_calibration)
        print("Calibration list has been constructed and totally {} data".format(len(self.nc_calibration)))
    
    def __call__(self, image):
        image = self.transform(image)
        image = image.to(self.device)
        image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        with torch.no_grad():
            x, mu, logvar = self.net(image)
            nc = self.mse(image, x).item()
        p = (100 - stats.percentileofscore(self.nc_calibration, nc))/float(100)
        return p
        

    

