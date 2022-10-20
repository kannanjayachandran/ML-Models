import torch
import time


# Setting device to Cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 32 is the batch size
matrix_size = 32*250

# setting Matrix
x = torch.randn(matrix_size, matrix_size)
y = torch.randn(matrix_size, matrix_size)

# testing CPU
print('------- CPU Speed -------')

start = time.time()
result = torch.matmul(x, y)
print(time.time() - start, "Seconds", end=" ")

print(f'Using {result.device}\n')

# setting device as GPU
x_gpu = x.to(device)
y_gpu = y.to(device)

# stoping CPU for executing code in GPU
torch.cuda.synchronize()

print('------- GPU Speed -------')

# testing GPU
for i in range(3):

    start = time.time()
    result_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    print((time.time() - start), 'Seconds.')

print(f'Using {result_gpu.device}')
