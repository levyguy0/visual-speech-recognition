# import torch
import os

# from train import LipReader

# model = LipReader()
# model.load_state_dict(torch.load("best_model.pth"))
print([s for s in os.listdir("data/videos") if s != ".DS_Store"])
