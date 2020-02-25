"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-22 23:05:11
@modify date 2020-02-22 23:05:11
@desc The codes are used for out-of-distribution detection
"""

import os
import csv
from PIL import Image
from scripts.icad import ICAD
from scripts.martingales import SMM
import numpy as np


train_path = "./data/training/"
icad = ICAD(train_path)
N = 10

data_path = "./data/test/out"
with open(os.path.join(data_path, "label.csv")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for index, row in enumerate(csv_reader):
        image_path = os.path.join(data_path, str(index)+".png")
        image = Image.open(image_path).convert("RGB")
        smm = SMM(N)
        for i in range(N):
            p = icad(image)
            m = smm(p)
        print(index, p, np.log(m))

