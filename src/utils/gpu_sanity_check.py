import torch, time
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
print("Device:", torch.cuda.get_device_name(0))
x = torch.randn(4096,4096, device='cuda')
t0=time.time(); y = x @ x.t(); torch.cuda.synchronize();
print("GEMM ms:", (time.time()-t0)*1e3)
