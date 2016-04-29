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

/* population count (number of one bits) from http://aggregate.org/MAGIC/ */
inline static uniform unsigned int ones (uniform unsigned int x)
{
  x -= ((x >> 1) & 0x55555555);
  x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
  x = (((x >> 4) + x) & 0x0f0f0f0f);
  x += (x >> 8);
  x += (x >> 16);
  return (x & 0x0000003f);
}

/* leading zero count from http://aggregate.org/MAGIC/ */
inline static uniform unsigned int lzc (uniform unsigned int x)
{
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return (32 - ones(x));
}

/* generalised leading zero count as required by the radix tree algorithm */
inline static uniform int delta (uniform int i, uniform unsigned int codei, uniform int j, uniform int n, uniform unsigned int code[])
{
  if (j < 0 || j > n-1) return -1;

  uniform unsigned int codej = code[j];

  if (codei == codej) return 32 + lzc (i ^ j);
  else return lzc (codei ^ codej);
}

/* http://stackoverflow.com/questions/14579920/fast-sign-of-integer-in-c */
inline static uniform int sign (uniform int x)
{
  return (uniform int)(x > 0) - (uniform int)(x < 0);
}

/* binary space partitioning radix tree */
struct radix_tree
{
  uniform int parent;
  uniform int first;
  uniform int size;
  uniform int split;

  uniform REAL coord;
  uniform int dimension;
};

/* find minimum coord in re-order-ed [start,end) range */
inline static uniform REAL mincoord (uniform REAL coord[], uniform int order[], uniform int start, uniform int end)
{
  uniform REAL ret = REAL_MAX;

  for (uniform int i = start; i < end; i ++)
  {
    uniform REAL val = coord[order[i]];
    if (val < ret) ret = val;
  }

  return ret;
}

/* from https://research.nvidia.com/publication/maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees */
task void radix_tree_create (uniform int span, uniform int n, uniform unsigned int code[],
  uniform radix_tree rtree[], uniform int order[], uniform REAL * uniform point[3], uniform int cutoff)
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n-1: start+span;

  for (uniform int i = start; i < end; i ++)
  {
    uniform int codei = code[i];

    uniform int d = sign (delta(i, codei, i+1, n, code) - delta(i, codei, i-1, n, code));

    uniform int dmin = delta (i, codei, i-d, n, code);

    uniform int lmax = 2;

    while (delta (i, codei, i+lmax*d, n, code) > dmin) lmax *= 2;

    uniform int l = 0;

    for (uniform int t = lmax/2; t >= 1; t /= 2)
    {
      if (delta (i, codei, i+(l+t)*d, n, code) > dmin) l = l+t;
    }

    uniform int j = i+l*d;

    rtree[i].first = d > 0 ? i : j;
    rtree[i].size = l+1;

    if (rtree[i].size <= cutoff) /* create terminal node and copy particle data */
    {
      rtree[i].split = 0;
      rtree[i].coord = 0.0;
      rtree[i].dimension = -1; /* mark as terminal node */
    }
    else
    {
      uniform int dnode = delta (i, codei, j, n, code);

      uniform int s = 0;

      for (uniform int t = (l+1)/2;; t = (t+1)/2)
      {
	if (delta (i, codei, i+(s+t)*d, n, code) > dnode) s = s+t;
	if (t == 1) break;
      }

      rtree[i].split = i+s*d+min(d,0);

      if (rtree[i].first != rtree[i].split) /* not left leaf => left node */
      {
	rtree[rtree[i].split].parent = i; /* left node parent */
      }

      if (rtree[i].first+rtree[i].size-1 != rtree[i].split+1) /* not right leaf => right node */
      {
	rtree[rtree[i].split+1].parent = i; /* right node parent */
      }

      uniform int dimension = (dnode-2)%3;

      rtree[i].coord = mincoord (point[dimension], order, rtree[i].split+1, rtree[i].first+rtree[i].size);
      rtree[i].dimension = dimension;
    }
  }

  if (taskIndex == 0)
  {
    rtree[0].parent = -1; /* nil root's parent */

    if (n == 1) /* handle the signe sphere case */
    {
      rtree[0].first = 0;
      rtree[0].size = 1;
      rtree[0].dimension = -1;
    }
  }
}

/* calculate size of the radix tree */
static void radix_tree_size (uniform radix_tree rtree[], uniform int node, uniform int * uniform size)
{
  if (rtree[node].dimension >= 0) /* node */
  {
    (*size) += 2; /* two potential siblings */

    uniform int j = rtree[node].split;

    if (rtree[node].first != j) /* not left leaf */
      radix_tree_size (rtree, j, size);

    if ((rtree[node].first+rtree[node].size-1) != (j+1)) /* not right leaf */
      radix_tree_size (rtree, j+1, size);
  }
}

/* create paritioning tree from the radix tree and copy particles into it */
static void partitioning_tree_create (uniform radix_tree rtree[], uniform int rnode,
  uniform partitioning ptree[], uniform int pnode, uniform int * uniform i, uniform int * uniform count)
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
      partitioning_tree_create (rtree, j, ptree, ptree[pnode].left, i, count);
    else /* left leaf */
    {
      uniform int left = ptree[pnode].left;
      ptree[left].coord = 0.0;
      ptree[left].dimension = -1;
      ptree[left].left = ptree[left].right = -1;
      ptree[left].rank = -1;
      ptree[left].size = 0;
      (*count) ++;
    }

    if ((rtree[rnode].first+rtree[rnode].size-1) != (j+1)) /* not right leaf */
      partitioning_tree_create (rtree, j+1, ptree, ptree[pnode].right, i, count);
    else /* right leaf */
    {
      uniform int right = ptree[pnode].right;
      ptree[right].coord = 0.0;
      ptree[right].dimension = -1;
      ptree[right].left = ptree[right].right = -1;
      ptree[right].rank = -1;
      ptree[right].size = 0;
      (*count) ++;
    }
  }
  else /* leaf */
  {
    ptree[pnode].left = ptree[pnode].right = -1;
    ptree[pnode].rank = -1;
    ptree[pnode].size = 0;
    (*count) ++;
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
    if (point[d][i] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_create */
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

/* create partitioning tree */
export uniform partitioning * uniform partitioning_create (uniform int ntasks, uniform int n, uniform REAL * uniform point[3],
  uniform int cutoff, uniform int * uniform tree_size, uniform int * uniform leaf_count)
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;

  uniform uint * uniform code = uniform new uniform uint [n];

  uniform int * uniform order = uniform new uniform int [n];

  morton_ordering (ntasks, n, point, code, order);

  uniform radix_tree * uniform rtree = uniform new uniform radix_tree [n];

  launch[num] radix_tree_create (span, n, code, rtree, order, point, cutoff);
  sync;

  *tree_size = 1;

  radix_tree_size (rtree, 0, tree_size);

  uniform partitioning * uniform ptree = uniform new uniform partitioning[*tree_size];

  uniform int i = 0;

  *leaf_count = 0;

  partitioning_tree_create (rtree, 0, ptree, 0, &i, leaf_count);

  delete code;
  delete order;
  delete rtree;

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
    if (point[d] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_create */
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
  uniform REAL lo[], uniform REAL hi[], uniform int ranks[], uniform int * uniform count)
{
  uniform int d = ptree[node].dimension;

  if (d >= 0) /* node */
  {
    if (lo[d] < ptree[node].coord) /* "<" is congruent with the selection of coord in radix_tree_create */
      partitioning_box_assign (ptree, ptree[node].left, lo, hi, ranks, count);
    if (hi[d] > ptree[node].coord)
      partitioning_box_assign (ptree, ptree[node].right, lo, hi, ranks, count);
  }
  else /* leaf */
  {
    uniform int i, r = ptree[node].rank;
    
    for (i = 0; i < (*count); i ++)
    {
      if (ranks[i] == r) break;
    }

    if (i == (*count))
    {
      ranks[i] = r;
      (*count) ++;
    }
  }
}

/* destroy partitioning tree */
export void partitioning_destroy (uniform partitioning * uniform ptree)
{
  delete ptree;
}
