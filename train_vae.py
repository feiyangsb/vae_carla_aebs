from scripts.vae_trainer import VAETrainer

data_path = "./data/training/"
trainer = VAETrainer(data_path, 350)
trainer.fit()
trainer.save_model()
