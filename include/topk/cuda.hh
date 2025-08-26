/* stub */

namespace apex::cuda {

/* nested library tree data structure, from list[list[list[int]]] */
using LibraryTree = std::vector<std::vector<std::vector<int64_t>>>;

std::tuple<torch::Tensor, torch::Tensor>
topk(
  torch::Tensor contribs,
  torch::Tensor quadratic_ids,
  torch::Tensor objective_ids,
  torch::Tensor constraint_ids,
  torch::Tensor constraint_bounds,
  const LibraryTree& libtree_data,
  const int64_t topk,
  bool descending,
  // 2^30 = 1073741824, should adjust for GPU memory,
  // but can be in 64-bit territory
  const int64_t default_chunk_size = 1 << 30
) {
  return {torch::Tensor(), torch::Tensor()};
}

} /* namespace apex::cuda */
