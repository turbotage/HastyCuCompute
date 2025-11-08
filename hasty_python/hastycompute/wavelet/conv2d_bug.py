"""
Minimal reproducible example for PyTorch CUDA precision bug with large filters.

Bug Description:
- conv2d and conv_transpose2d produce incorrect results on CUDA with filters >= 10x10
- CPU results are correct
- 1D and 3D convolutions work correctly on CUDA
- Only 2D convolutions are affected

Environment where bug was confirmed:
- PyTorch 2.8.0 + CUDA 12.9
- PyTorch 2.9.0 + CUDA 13.0

Expected behavior:
- CPU and CUDA should produce identical results (within floating-point tolerance ~1e-6)

Actual behavior:
- CPU vs CUDA differences are ~1e-2 to 2e-2 for filters >= 10x10
- Filters <= 8x8 work correctly (diff ~0)
"""

import torch
import torch.nn.functional as F


def test_conv2d_precision():
    """Test conv2d precision on CPU vs CUDA for various filter sizes."""
    print("="*80)
    print("PyTorch CUDA Bug Report: conv2d precision issue with large filters")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - cannot demonstrate bug")
        return
    
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 100, 100)
    
    print("\nTest 1: conv2d with different filter sizes")
    print("-" * 80)
    print(f"{'Filter Size':<15} {'Max CPU vs CUDA Diff':<25} {'Status'}")
    print("-" * 80)
    
    for size in [4, 6, 8, 10, 12, 14, 16, 18]:
        filter_tensor = torch.randn(1, 1, size, size)
        
        # CPU computation
        out_cpu = F.conv2d(input_tensor, filter_tensor, padding=size//2)
        
        # CUDA computation
        out_cuda = F.conv2d(input_tensor.cuda(), filter_tensor.cuda(), padding=size//2)
        
        # Compare
        diff = torch.max(torch.abs(out_cpu - out_cuda.cpu())).item()
        status = '✓ OK' if diff < 1e-5 else '✗ FAIL'
        print(f'{size}x{size:<11} {diff:<25.2e} {status}')
    
    print("\n" + "="*80)


def test_conv_transpose2d_precision():
    """Test conv_transpose2d precision on CPU vs CUDA."""
    print("\nTest 2: conv_transpose2d with 12x12 filters")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    
    # Typical scenario: upsampling in wavelet reconstruction
    input_tensor = torch.randn(1, 4, 37, 37)
    filter_12x12 = torch.randn(4, 1, 12, 12)
    
    # CPU computation
    out_cpu = F.conv_transpose2d(input_tensor, filter_12x12, stride=2)
    
    # CUDA computation
    out_cuda = F.conv_transpose2d(input_tensor.cuda(), filter_12x12.cuda(), stride=2)
    
    # Compare
    diff = torch.max(torch.abs(out_cpu - out_cuda.cpu())).item()
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Filter shape: {filter_12x12.shape}")
    print(f"Output shape: {out_cpu.shape}")
    print(f"Max CPU vs CUDA diff: {diff:.2e}")
    
    if diff > 1e-5:
        print("✗ FAIL - Large precision error on CUDA")
    else:
        print("✓ OK - Results match within tolerance")
    
    print("="*80)


def test_1d_vs_2d_vs_3d():
    """Demonstrate that only 2D convolutions are affected."""
    print("\nTest 3: Comparing 1D, 2D, and 3D convolutions with 12-coefficient filters")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    
    # Test 1D conv_transpose1d
    input_1d = torch.randn(1, 4, 37)
    filter_1d = torch.randn(4, 1, 12)
    out_1d_cpu = F.conv_transpose1d(input_1d, filter_1d, stride=2)
    out_1d_cuda = F.conv_transpose1d(input_1d.cuda(), filter_1d.cuda(), stride=2)
    diff_1d = torch.max(torch.abs(out_1d_cpu - out_1d_cuda.cpu())).item()
    
    # Test 2D conv_transpose2d
    input_2d = torch.randn(1, 4, 37, 37)
    filter_2d = torch.randn(4, 1, 12, 12)
    out_2d_cpu = F.conv_transpose2d(input_2d, filter_2d, stride=2)
    out_2d_cuda = F.conv_transpose2d(input_2d.cuda(), filter_2d.cuda(), stride=2)
    diff_2d = torch.max(torch.abs(out_2d_cpu - out_2d_cuda.cpu())).item()
    
    # Test 3D conv_transpose3d
    input_3d = torch.randn(1, 4, 16, 16, 16)
    filter_3d = torch.randn(4, 1, 12, 12, 12)
    out_3d_cpu = F.conv_transpose3d(input_3d, filter_3d, stride=2)
    out_3d_cuda = F.conv_transpose3d(input_3d.cuda(), filter_3d.cuda(), stride=2)
    diff_3d = torch.max(torch.abs(out_3d_cpu - out_3d_cuda.cpu())).item()
    
    print(f"{'Dimension':<15} {'Max CPU vs CUDA Diff':<25} {'Status'}")
    print("-" * 80)
    print(f"{'1D (12 coeffs)':<15} {diff_1d:<25.2e} {'✓ OK' if diff_1d < 1e-4 else '✗ FAIL'}")
    print(f"{'2D (12x12)':<15} {diff_2d:<25.2e} {'✓ OK' if diff_2d < 1e-4 else '✗ FAIL'}")
    print(f"{'3D (12x12x12)':<15} {diff_3d:<25.2e} {'✓ OK' if diff_3d < 1e-4 else '✗ FAIL'}")
    
    print("\n" + "="*80)
    print("EXPECTED RESULT: Only 2D should fail, 1D and 3D should pass")
    print("="*80)


def main():
    """Run all tests to demonstrate the bug."""
    test_conv2d_precision()
    test_conv_transpose2d_precision()
    test_1d_vs_2d_vs_3d()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("This bug affects:")
    print("  - F.conv2d with filters >= 10x10")
    print("  - F.conv_transpose2d with filters >= 10x10")
    print("  - Only 2D operations (1D and 3D work correctly)")
    print("\nThis bug does NOT affect:")
    print("  - CPU computations (always correct)")
    print("  - Filters <= 8x8 (work on CUDA)")
    print("  - 1D/3D convolutions (work on CUDA)")
    print("\nImpact:")
    print("  - Wavelet transforms with long-filter wavelets (db6, db8, sym8, coif2, coif3)")
    print("  - Any application using 2D convolution with filters >= 10x10 on CUDA")
    print("  - Precision errors are ~1e-2 to 2e-2, which is unacceptable for many applications")
    print("="*80)


if __name__ == "__main__":
    main()
