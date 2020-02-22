"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-22 22:06:07
@modify date 2020-02-22 22:06:07
@desc The codes is used to compare the reconstructed image and the original image
"""
from scripts.network import VAE
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

image_path = "./data/training/setting_1/19/50.png"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = Image.open(image_path).convert("RGB")
image.show()
image = transform(image)

vae = VAE()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device)
vae.load_state_dict(torch.load("./model/vae.pt", map_location=device))
vae.eval()

image = image.view([1, image.shape[0], image.shape[1], image.shape[2]])
x, mu, logvar = vae(image)
print(x.shape)
reconstructed_image = x[0].cpu().data.numpy()
reconstructed_image = np.rollaxis(reconstructed_image, 0, 3)
reconstructed_image = Image.fromarray(np.uint8(reconstructed_image*255.0))
reconstructed_image.show()

