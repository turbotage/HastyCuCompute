import torch
import time

io = torch.rand(640, 640, 640, dtype=torch.complex64)

for i in range(5):
    start = time.time()
    torch.save(io, '/home/turbotage/Documents/io.pt')
    end = time.time()
    timestamp = end - start
    print('Time to save:', timestamp, ' ', (io.element_size() * io.nelement() / (1024 * 1024)) / timestamp, 'MB/s')

    start = time.time()
    io = torch.load('/home/turbotage/Documents/io.pt')
    end = time.time()
    timestamp = end - start
    print('Time to load:', timestamp, ' ', (io.element_size() * io.nelement() / (1024 * 1024)) / timestamp, 'MB/s')

    





