#include <torch/script.h>
#include <torch/extension.h>

#include <cuda.h>

#include <topk/cuda.hh>

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("topk", apex::cuda::topk);
}
