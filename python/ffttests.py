import cupy as cp
import numpy as np
from pyvkfft.cuda import VkFFTApp

def benchmark_fft(N, iters=5):
    npts = N**3
    print(f"\n=== {N}^3 complex FFT ({npts:,} points) ===")

    # allocate data
    x = cp.random.random((N, N, N), dtype=cp.float32) + 1j * cp.random.random((N, N, N), dtype=cp.float32)

    # --- cuFFT (via CuPy) ---
    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    # warmup
    _ = cp.fft.fftn(x)
    cp.cuda.Stream.null.synchronize()

    start.record()
    for _ in range(iters):
        _ = cp.fft.fftn(x)
        _ = cp.fft.ifftn(x)
    stop.record()
    stop.synchronize()
    time_cufft = cp.cuda.get_elapsed_time(start, stop) / iters
    print(f"cuFFT avg time: {time_cufft:.3f} ms")

    # --- VkFFT (via pyvkfft) ---
    fft = VkFFTApp(x.shape, dtype=np.complex64)
    y = cp.empty_like(x)

    # warmup
    fft.fft(x, y)
    fft.ifft(y, x)
    cp.cuda.Stream.null.synchronize()

    start.record()
    for _ in range(iters):
        fft.fft(x, y)
        fft.ifft(y, x)
    stop.record()
    stop.synchronize()
    time_vkfft = cp.cuda.get_elapsed_time(start, stop) / iters
    print(f"VkFFT avg time: {time_vkfft:.3f} ms")


if __name__ == "__main__":
    for N in [256, 512, 640]:
        try:
            benchmark_fft(N)
        except cp.cuda.memory.OutOfMemoryError:
            print(f"Skipping {N}^3 (not enough GPU memory)")