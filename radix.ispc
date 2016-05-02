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

typedef unsigned int uint;

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
task void radix_tree_task (uniform int span, uniform int n, uniform unsigned int code[],
  uniform radix_tree tree[], uniform int order[], uniform REAL * uniform point[3], uniform int cutoff)
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

    tree[i].first = d > 0 ? i : j;
    tree[i].size = l+1;

    if (tree[i].size <= cutoff) /* create terminal node and copy particle data */
    {
      tree[i].split = 0;
      tree[i].coord = 0.0;
      tree[i].dimension = -1; /* mark as terminal node */
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

      tree[i].split = i+s*d+min(d,0);

      if (tree[i].first != tree[i].split) /* not left leaf => left node */
      {
	tree[tree[i].split].parent = i; /* left node parent */
      }

      if (tree[i].first+tree[i].size-1 != tree[i].split+1) /* not right leaf => right node */
      {
	tree[tree[i].split+1].parent = i; /* right node parent */
      }

      uniform int dimension = (dnode-2)%3;

      tree[i].coord = mincoord (point[dimension], order, tree[i].split+1, tree[i].first+tree[i].size);
      tree[i].dimension = dimension;
    }
  }

  if (taskIndex == 0)
  {
    tree[0].parent = -1; /* nil root's parent */

    if (n == 1) /* handle the signe sphere case */
    {
      tree[0].first = 0;
      tree[0].size = 1;
      tree[0].dimension = -1;
    }
  }
}

/* calculate size of the radix tree */
static void radix_tree_size (uniform radix_tree tree[], uniform int node, uniform int * uniform size)
{
  if (tree[node].dimension >= 0) /* node */
  {
    (*size) += 2; /* two potential siblings */

    uniform int j = tree[node].split;

    if (tree[node].first != j) /* not left leaf */
      radix_tree_size (tree, j, size);

    if ((tree[node].first+tree[node].size-1) != (j+1)) /* not right leaf */
      radix_tree_size (tree, j+1, size);
  }
}

/* create radix tree */
export uniform radix_tree * uniform radix_tree_create (uniform int ntasks, uniform int n,
  uniform REAL * uniform point[3], uniform int cutoff, uniform int * uniform tree_size)
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;

  uniform uint * uniform code = uniform new uniform uint [n];

  uniform int * uniform order = uniform new uniform int [n];

  morton_ordering (ntasks, n, point, code, order);

  uniform radix_tree * uniform tree = uniform new uniform radix_tree [n];

  launch[num] radix_tree_task (span, n, code, tree, order, point, cutoff);
  sync;

  *tree_size = 1;

  radix_tree_size (tree, 0, tree_size);

  delete code;
  delete order;

  return tree;
}

/* destroy radix tree */
export void radix_tree_destroy (uniform radix_tree * uniform tree)
{
  delete tree;
}
