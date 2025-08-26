/* cpu implementation of apex topk.
 * Written by Bradley Worley <brad@atomwise.com>.
 *
 * Copyright (c) 2025 Atomwise. All Rights Reserved.
 * Confidential. Restricted to the Atomwise/NVIDIA collaboration.
 */

#pragma once

#include <array>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

namespace apex::cpu {

/* nested library tree data structure, from list[list[list[int]]] */
using LibraryTree = std::vector<std::vector<std::vector<int64_t>>>;

/* max_groups: maximum number of component r-groups per reaction.
 * max_terms: maximum second-dim size of the contributions tensor.
 */
static constexpr int max_groups = 4;
static constexpr int max_terms = 256;

/* struct holding each search result, identifies the molecule. */
using index_t = int32_t;
struct result_t {
  float slack{std::numeric_limits<float>::min()};
  float obj{std::numeric_limits<float>::min()};

  index_t reaction{-1};
  std::array<index_t, max_groups> groups{-1, -1, -1, -1};
};

/* struct to compare search results by score; builds a min-heap. */
struct result_greater {
  bool operator()(const result_t& first, const result_t& second) {
    if (first.slack == second.slack)
      return first.obj > second.obj;
    else
      return first.slack > second.slack;
  }
};

/* struct holding intermediate results for a single term.
 *  accum: accumulator for sums of r-group contributions.
 *  constraint_accum: accumulator for sums of constraint terms.
 *  lower: lower bound of constraints.
 *  upper: upper bound of constraints.
 *  dest: constraint destination index, or -1 if no destination.
 *  quadratic: when true, the term must be squared.
 *  objective: when true, the term is an objective.
 *  constraint: when true, the term is a constraint.
 */
struct scratch_t {
  double accum{0.0};
  double constraint_accum{0.0};
  float lower{0.0f};
  float upper{0.0f};
  index_t dest{-1};
  bool quadratic{false};
  bool objective{false};
  bool constraint{false};
};

/* simplest possible rectified linear unit. */
inline float relu(float x) {
  return x > 0.0f ? x : 0.0f;
}

/* min-heap priority queue used to exhaustively search a library tree.
 */
class result_queue
: public std::priority_queue<result_t, std::vector<result_t>, result_greater> {
public:
  /* constructor. */
  result_queue(
    std::size_t topk,
    const torch::TensorAccessor<float, 2>& contribs_accessor,
    const torch::TensorAccessor<int64_t, 1>& quadratic_accessor,
    const torch::TensorAccessor<int64_t, 1>& objective_accessor,
    const torch::TensorAccessor<int64_t, 2>& constraint_id_accessor,
    const torch::TensorAccessor<float, 2>& constraint_bound_accessor
  )
  : priority_queue(),
    max_size(topk),
    num_terms(contribs_accessor.size(/*dim=*/1)),
    contribs(contribs_accessor),
    scratch() {
    /* initialize the scratch space. */
    const int64_t n_quadratic = quadratic_accessor.size(/*dim=*/0);
    for (int64_t index = 0; index < n_quadratic; ++index)
      scratch[quadratic_accessor[index]].quadratic = true;

    const int64_t n_objective = objective_accessor.size(/*dim=*/0);
    for (int64_t index = 0; index < n_objective; ++index)
      scratch[objective_accessor[index]].objective = true;

    const int64_t n_constraint = constraint_id_accessor.size(/*dim=*/1);
    for (int64_t index = 0; index < n_constraint; ++index) {
      const int64_t dest_index = constraint_id_accessor[0][index];
      const int64_t src_index = constraint_id_accessor[1][index];

      scratch[dest_index].constraint = true;
      scratch[dest_index].lower = constraint_bound_accessor[0][dest_index];
      scratch[dest_index].upper = constraint_bound_accessor[1][dest_index];

      scratch[src_index].dest = static_cast<index_t>(dest_index);
    }
  }

  /* tail condition of the push_reaction<> template, below. */
  template<int Position>
  void push_reaction(result_t& result) {
    /* zero out the constraint accumulators. */
    for (unsigned int t = 0; t < num_terms; ++t)
      scratch[t].constraint_accum = 0.0;

    /* compute the objective and the constraint sums. */
    for (unsigned int t = 0; t < num_terms; ++t) {
      const auto& term = scratch[t];

      const double value = term.quadratic
                         ? term.accum * term.accum
                         : term.accum;

      if (term.dest >= 0)
        scratch[term.dest].constraint_accum += value;

      if (term.objective)
        result.obj += value;
    }

    /* compute the constraint violations. */
    for (unsigned int t = 0; t < num_terms; ++t) {
      const auto& term = scratch[t];
      if (term.constraint)
        result.slack -= relu(term.lower - term.constraint_accum)
                      + relu(term.constraint_accum - term.upper);
    }

    if (size() < max_size) {
      push(result);
    }
    else if (comp(result, top())) { /* result > top() */
      pop();
      push(result);
    }
  }

  /* push all results from a reaction onto the queue. */
  template<int Position = 0, typename First, typename... Rest>
  void push_reaction(
    const result_t& partial,
    const First& rgroup_ids,
    const Rest&... other_ids
  ) {
    static_assert(Position < max_groups);

    for (const auto r : rgroup_ids) {
      result_t updated{partial};
      updated.groups[Position] = static_cast<index_t>(r);

      if constexpr (Position == 0) {
        for (unsigned int t = 0; t < num_terms; ++t)
          scratch[t].accum = contribs[r][t];
      }
      else {
        for (unsigned int t = 0; t < num_terms; ++t)
          scratch[t].accum += contribs[r][t];
      }

      push_reaction<Position + 1, Rest...>(updated, other_ids...);

      for (unsigned int t = 0; t < num_terms; ++t)
        scratch[t].accum -= contribs[r][t];
    }
  }

  /* push all results from a library onto the queue. */
  void push_library(const LibraryTree& libtree_data) {
    index_t index{0};
    for (const auto& rxn : libtree_data) {
      const int64_t num_groups = rxn.size();
      if (num_groups == 2)
        push_reaction({0.0f, 0.0f, index}, rxn[0], rxn[1]);
      else if (num_groups == 3)
        push_reaction({0.0f, 0.0f, index}, rxn[0], rxn[1], rxn[2]);
      else if (num_groups == 4)
        push_reaction({0.0f, 0.0f, index}, rxn[0], rxn[1], rxn[2], rxn[3]);
      else
        throw std::length_error{"invalid number of components in reaction"};

      ++index;
    }
  }

  /* return score and index tensors; consumes the queue contents. */
  std::tuple<torch::Tensor, torch::Tensor> as_tensors(bool descending) {
    const int64_t count = size();

    const auto f32 = torch::dtype(torch::kFloat32);
    auto scores = torch::empty({count, 2}, /*options=*/f32);
    auto scores_accessor = scores.accessor<float, 2>();

    const auto i64 = torch::dtype(torch::kInt64);
    auto ids = torch::empty({count, max_groups + 1}, /*options=*/i64);
    auto ids_accessor = ids.accessor<int64_t, 2>();

    int64_t index = descending ? count - 1 : 0;
    while (!empty()) {
      const auto& result = top();

      scores_accessor[index][0] = result.slack;
      scores_accessor[index][1] = result.obj;

      ids_accessor[index][0] = static_cast<int64_t>(result.reaction);

      for (int64_t g = 0; g < max_groups; ++g)
        ids_accessor[index][g + 1] = static_cast<int64_t>(result.groups[g]);

      pop();
      descending ? --index : ++index;
    }

    return {scores, ids};
  }

private:
  const std::size_t max_size, num_terms;
  const torch::TensorAccessor<float, 2> contribs;
  std::array<scratch_t, max_terms> scratch;
};

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
  int64_t chunk_size  /* unused on cpu. */
) {
  /* check a few things for safety... */
  const int64_t max_contribs = std::numeric_limits<index_t>::max();

  if (contribs.size(/*dim=*/0) >= max_contribs)
    throw std::length_error{"exceeded maximum number of contributions"};

  if (contribs.size(/*dim=*/1) >= max_terms)
    throw std::length_error{"exceeded maximum number of score terms"};

  if (quadratic_ids.lt(0).any().item<bool>() ||
      quadratic_ids.ge(max_terms).any().item<bool>())
    throw std::length_error{"quadratic indices are out of bounds"};

  if (constraint_ids.lt(0).any().item<bool>() ||
      constraint_ids.ge(max_terms).any().item<bool>())
    throw std::length_error{"constraint indices are out of bounds"};

  /* instantiate the priority queue and search the library. */
  result_queue queue(
    topk,
    contribs.accessor<float, 2>(),
    quadratic_ids.accessor<int64_t, 1>(),
    objective_ids.accessor<int64_t, 1>(),
    constraint_ids.accessor<int64_t, 2>(),
    constraint_bounds.accessor<float, 2>()
  );
  queue.push_library(libtree_data);
  return queue.as_tensors(descending);
}

} /* namespace apex::cpu */
