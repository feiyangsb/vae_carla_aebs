"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-22 11:21:13
@modify date 2020-02-22 11:21:13
@desc The codes is used to visualize the latent space of VAE
"""

from scripts.network import VAE
import torch
from scripts.data_loader import CarlaAEBSDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

vae = VAE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
vae.load_state_dict(torch.load("./model/vae.pt", map_location=device))
vae.eval()

try:
    mu_list = np.load("./data/visualization/X_2d.npy")
    gt_distance_list = np.load("./data/visualization/gt_dist.npy")
except:
    mu_list = []
    gt_distance_list = []
    dataset = CarlaAEBSDataset("./data/training", "calibration")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for idx, (inputs, targets) in enumerate(data_loader):
        print(idx)
        inputs = inputs.to(device)
        with torch.no_grad():
            x, mu, logvar = vae(inputs)
            mu_list.append(mu[0].cpu().data.numpy())
            gt_distance_list.append(targets[0].cpu().data.numpy())

    mu_list = np.asarray(mu_list)
    gt_distance_list = np.asarray(gt_distance_list)
    print("The prediction phase is done")
    tsne = MDS(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(mu_list)
    np.save("./data/visualization/X_2d.npy", X_2d)
    np.save("./data/visualization/gt_dist.npy", gt_distance_list)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=gt_distance_list[:,0], s=20)
plt.show()


    