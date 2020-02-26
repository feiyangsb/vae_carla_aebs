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
from scripts.detector import StatefulDetector
import numpy as np
import matplotlib.pyplot as plt


train_path = "./data/training/"
icad = ICAD(train_path)
N = 10

p_step_list = []
p_list = []
m_step_list = []
m_list = []
s_list = []
detector = StatefulDetector(10, 14)
data_path = "./data/test/in"
with open(os.path.join(data_path, "label.csv")) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for index, row in enumerate(csv_reader):
        image_path = os.path.join(data_path, str(index)+".png")
        image = Image.open(image_path).convert("RGB")
        smm = SMM(N)
        for i in range(N):
            p = icad(image)
            p_step_list.append(index)
            p_list.append(p)
            m = smm(p)
        m_step_list.append(index)
        m_list.append(np.log(m))
        s, _ = detector(np.log(m))
        s_list.append(s)

        print(index, p, np.log(m), s)

f, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(p_step_list, p_list, 'b.')
ax2.plot(m_step_list, m_list, 'b')
ax3.plot(m_step_list, s_list, 'b')
plt.show()

