import pickle
import os
import numpy as np

def pretrained_model_filename(freefermion_path):
    return os.path.join(freefermion_path, "params_van.pkl")

def ckpt_filename(epoch, path):
    return os.path.join(path, "epoch_%06d.pkl" % epoch)

def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_txt(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    data = np.array(data, dtype=np.float64)
    return data