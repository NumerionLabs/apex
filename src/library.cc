#include <torch/script.h>

TORCH_LIBRARY(apex, m) {
  m.def(
    "topk("
    " Tensor contribs,"
    " Tensor quadratic_ids,"
    " Tensor objective_ids,"
    " Tensor constraint_ids,"
    " Tensor constraint_bounds,"
    " int[][][] libtree_data,"
    " int topk,"
    " bool descending,"
    " int chunk_size"
    ") -> (Tensor, Tensor)"
  );
}
