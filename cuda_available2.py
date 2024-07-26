import torch
import os

def check_cuda():
    print("Torch CUDA available: ", torch.cuda.is_available())
    print("Torch CUDA version: ", torch.version.cuda)
    print("CUDA_VISIBLE_DEVICES: ", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))

if __name__ == "__main__":
    check_cuda()

