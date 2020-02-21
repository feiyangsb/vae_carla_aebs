"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-21 22:38:53
@modify date 2020-02-21 22:38:53
@desc main code to train the vae
"""

from scripts.vae_trainer import VAETrainer

data_path = "./data/training/"
trainer = VAETrainer(data_path, 350)
trainer.fit()
trainer.save_model()
