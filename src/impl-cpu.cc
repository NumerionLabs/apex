#include <torch/script.h>
#include <torch/extension.h>

#include <topk/cpu.hh>

TORCH_LIBRARY_IMPL(apex, CPU, m) {
  m.impl("topk", apex::cpu::topk);
}
