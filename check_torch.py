"""Verify the PyTorch installation and perform a real CUDA computation."""

import torch
import torchvision


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"Compiled architectures: {torch.cuda.get_arch_list()}")
    cpu = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    expected = cpu @ cpu.T
    print("CPU matrix multiplication: OK")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check WSL GPU access and the NVIDIA driver.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    gpu = cpu.to("cuda")
    result = (gpu @ gpu.T).cpu()
    torch.testing.assert_close(result, expected)
    torch.cuda.synchronize()
    print("CUDA matrix multiplication: OK")


if __name__ == "__main__":
    main()
