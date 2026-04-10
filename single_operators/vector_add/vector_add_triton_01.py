import triton
import triton.language as tl
import os
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, 
               y_ptr,
               output_ptr,
               n_elements,
               #promised to be a const value
               BLOCK_SIZE: tl.constexpr
               ):
    #like blockIdx.x in cuda
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    #there is no 'thread' in triton, at least we don't need to pay attention to it
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    #used to mask 'thread'
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
    
    
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    #cdiv means 'ceiling division', and meta is a config map
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    #[] like <<<>>> in cuda with only the first parameter 'Grid'
    #can be used where the function is labeled by @triton.jit
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='triton-01-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
    
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
target_dir = os.path.join(parent_dir, "vec_add_performance")
benchmark.run(show_plots=True, save_path=target_dir)