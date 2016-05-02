/*
The MIT License (MIT)

Copyright (c) 2015 Tomasz Koziara

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "macros.h"
#include "morton.h"
#include "radix.h"
#include "rcb.h"

typedef unsigned int uint;

struct partitioning /* partitioning tree */
{
  uniform REAL coord;
  uniform int dimension;
  uniform int left;
  uniform int right;

  uniform int rank; /* > 0 for leaves */
  uniform int size; /* leaf size */
};

/* create paritioning tree from the radix tree */
static void tree_create_radix (uniform radix_tree rtree[], uniform int rnode,
  uniform partitioning ptree[], uniform int pnode, uniform int * uniform i, uniform int * uniform leaf_count)
{
  ptree[pnode].coord = rtree[rnode].coord;
  ptree[pnode].dimension = rtree[rnode].dimension;

  if (rtree[rnode].dimension >= 0) /* node */
  {
    ptree[pnode].left = ++(*i);
    ptree[pnode].right = ++(*i);
    ptree[pnode].rank = -1;
    ptree[pnode].size = -1;

    uniform int j = rtree[rnode].split;

    if (rtree[rnode].first != j) /* not left leaf */
      tree_create_radix (rtree, j, ptree, ptree[pnode].left, i, leaf_count);
    else /* left leaf */
    {
      uniform int left = ptree[pnode].left;
      ptree[left].coord = 0.0;
      ptree[left].dimension = -1;
      ptree[left].left = ptree[left].right = -1;
      ptree[left].rank = -1;
      ptree[left].size = 0;
      (*leaf_count) ++;
    }

    if ((rtree[rnode].first+rtree[rnode].size-1) != (j+1)) /* not right leaf */
      tree_create_radix (rtree, j+1, ptree, ptree[pnode].right, i, leaf_count);
    else /* right leaf */
    {
      uniform int right = ptree[pnode].right;
      ptree[right].coord = 0.0;
      ptree[right].dimension = -1;
      ptree[right].left = ptree[right].right = -1;
      ptree[right].rank = -1;
      ptree[right].size = 0;
      (*leaf_count) ++;
    }
  }
  else /* leaf */
  {
    ptree[pnode].left = ptree[pnode].right = -1;
    ptree[pnode].rank = -1;
    ptree[pnode].size = 0;
    (*leaf_count) ++;
  }
}

/* create paritioning tree from the recurisve bisection tree */
static void tree_create_rcb (uniform rcb_tree rcbtree[], uniform int rcbnode,
  uniform partitioning ptree[], uniform int pnode, uniform int * uniform i, uniform int * uniform leaf_count)
{
  ptree[pnode].coord = rcbtree[rcbnode].coord;
  ptree[pnode].dimension = rcbtree[rcbnode].dimension;

  if (rcbtree[rcbnode].dimension >= 0) /* node */
  {
    ptree[pnode].left = ++(*i);
    ptree[pnode].right = ++(*i);
    ptree[pnode].rank = -1;
    ptree[pnode].size = -1;

    tree_create_rcb (rcbtree, rcbtree[rcbnode].left, ptree, ptree[pnode].left, i, leaf_count);
    tree_create_rcb (rcbtree, rcbtree[rcbnode].right, ptree, ptree[pnode].right, i, leaf_count);
  }
  else /* leaf */
  {
    ptree[pnode].left = ptree[pnode].right = -1;
    ptree[pnode].rank = -1;
    ptree[pnode].size = 0;
    (*leaf_count) ++;
  }
}

/* zero leaf data */
static void zero_leaves (uniform struct partitioning ptree[], uniform int node)
{
  if (ptree[node].dimension >= 0) /* node */
  {
    zero_leaves (ptree, ptree[node].left);
    zero_leaves (ptree, ptree[node].right);
  }
  else /* leaf */
  {
    ptree[node].size = 0;
  }
}

/* drop point down the partitioning tree */
static void drop_point (uniform partitioning ptree[], uniform int node, uniform int i, uniform REAL * uniform point[3])
{
  uniform int d = ptree[node].dimension;

  if (d >= 0) /* node */
  {
    if (point[d][i] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_task */
      drop_point (ptree, ptree[node].left, i, point);
    else drop_point (ptree, ptree[node].right, i, point);
  }
  else /* leaf */
  {
    atomic_add_global (&ptree[node].size, 1);
  }
}

/* store points at tree leaves */
task void store_points (uniform int span, uniform partitioning ptree[], uniform int n, uniform REAL * uniform point[3])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n: start+span;

  for (uniform int i = start; i < end; i ++)
  {
    drop_point (ptree, 0, i, point);
  }
}

/* assign ranks to partitioning tree leaves */
export void assign_ranks (uniform partitioning * uniform ptree, uniform int node,
  uniform int leaves_per_rank, uniform int * uniform remainder,
  uniform int * uniform leaf, uniform int * uniform rank)
{
  if (ptree[node].dimension >= 0) /* node */
  {
    assign_ranks (ptree, ptree[node].left, leaves_per_rank, remainder, leaf, rank);
    assign_ranks (ptree, ptree[node].right, leaves_per_rank, remainder, leaf, rank);
  }
  else /* leaf */
  {
    ptree[node].rank =  (*rank);

    (*leaf) ++;

    uniform int m = leaves_per_rank + ((*remainder) ? 1 : 0); /* remainder is distributed into initial ranks */

    if ((*leaf) == m) /* finished with this rank */
    {
      (*leaf) = 0; /* zero leaf per rank counter */

      (*rank) ++; /* increment rank */

      if ((*remainder)) (*remainder) --; /* if remainder left then decrement it */
    }
  }
}

/* create partitioning tree based on radix tree */
export uniform partitioning * uniform partitioning_create_radix (uniform int ntasks, uniform int n, uniform REAL * uniform point[3],
  uniform int cutoff, uniform int * uniform tree_size, uniform int * uniform leaf_count)
{
  uniform radix_tree * uniform rtree = radix_tree_create (ntasks, n, point, cutoff, tree_size);

  uniform partitioning * uniform ptree = uniform new uniform partitioning[*tree_size];

  uniform int i = 0;

  *leaf_count = 0;

  tree_create_radix (rtree, 0, ptree, 0, &i, leaf_count);

  radix_tree_destroy (rtree);

  return ptree;
}

/* create partitioning tree based on rcb tree */
export uniform partitioning * uniform partitioning_create_rcb (uniform int ntasks, uniform int n, uniform REAL * uniform point[3],
  uniform int cutoff, uniform int * uniform tree_size, uniform int * uniform leaf_count)
{
  uniform rcb_tree * uniform rcbtree = rcb_tree_create (ntasks, n, point, cutoff, tree_size);

  uniform partitioning * uniform ptree = uniform new uniform partitioning[*tree_size];

  uniform int i = 0;

  *leaf_count = 0;

  tree_create_rcb (rcbtree, 0, ptree, 0, &i, leaf_count);

  rcb_tree_destroy (rcbtree);

  return ptree;
}

/* allocate partitioning tree memory */
export uniform partitioning * uniform partitioning_alloc (uniform int tree_size)
{
  uniform partitioning * uniform ptree = uniform new uniform partitioning[tree_size];

  return ptree;
}

/* assign ranks to partitioning tree leaves */
export void partitioning_assign_ranks (uniform partitioning * uniform ptree, uniform int leaves_per_rank, uniform int remainder)
{
  uniform int leaf = 0;

  uniform int rank = 0;

  assign_ranks (ptree, 0, leaves_per_rank, &remainder, &leaf, &rank);
}

/* store points in the partitioning tree leaves */
export void partitioning_store (uniform int ntasks, uniform partitioning * uniform ptree, uniform int n, uniform REAL * uniform point[3])
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;

  zero_leaves (ptree, 0);

  launch [num] store_points (n/num, ptree, n, point);
}

/* assign leaf rank to a point */
export uniform int partitioning_point_assign (uniform partitioning ptree[], uniform int node, uniform REAL point[])
{
  uniform int d = ptree[node].dimension;

  if (d >= 0) /* node */
  {
    if (point[d] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_task */
      return partitioning_point_assign (ptree, ptree[node].left, point);
    else return partitioning_point_assign (ptree, ptree[node].right, point);
  }
  else /* leaf */
  {
    return ptree[node].rank;
  }
}

/* assign leaf ranks to a box */
export void partitioning_box_assign (uniform partitioning ptree[], uniform int node,
  uniform REAL lo[], uniform REAL hi[], uniform int ranks[], uniform int * uniform rank_count)
{
  uniform int d = ptree[node].dimension;

  if (d >= 0) /* node */
  {
    if (lo[d] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_task */
      partitioning_box_assign (ptree, ptree[node].left, lo, hi, ranks, rank_count);
    if (hi[d] > ptree[node].coord)
      partitioning_box_assign (ptree, ptree[node].right, lo, hi, ranks, rank_count);
  }
  else /* leaf */
  {
    uniform int i, r = ptree[node].rank;
    
    for (i = 0; i < (*rank_count); i ++)
    {
      if (ranks[i] == r) break;
    }

    if (i == (*rank_count))
    {
      ranks[i] = r;
      (*rank_count) ++;
    }
  }
}

/* destroy partitioning tree */
export void partitioning_destroy (uniform partitioning * uniform ptree)
{
  delete ptree;
}
