/*
The MIT License (MIT)

Copyright (c) 2016 EDF Energy

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

/* Contributors: Tomasz Koziara */

#include "macros.h"
#include "morton.h"
#include "rcb.h"

static void rcb_tree_size (uniform int n, uniform int cutoff, uniform int * uniform tree_size)
{
  if (n > cutoff)
  {
    (*tree_size) += 2; /* two potential siblings */

    rcb_tree_size (n/2, cutoff, tree_size);
    rcb_tree_size (n-n/2, cutoff, tree_size);
  }
}

static void rcb_tree_init (uniform int n, uniform int cutoff, uniform rcb_tree tree[], uniform int node, uniform int * uniform i)
{
  if (n > cutoff) /* node */
  {
    tree[node].dimension = 3; /* mark as node node; actual dimension 0,1 or 2 will be determined in rcb_tree_task */
    tree[node].left = ++(*i);
    tree[node].right = ++(*i);

    rcb_tree_init (n/2, cutoff, tree, tree[node].left, i);
    rcb_tree_init (n-n/2, cutoff, tree, tree[node].right, i);
  }
  else /* leaf */
  {
    tree[node].dimension = -1; /* mark as leaf */
    tree[node].left = tree[node].right = -1;
  }
}

static void leaf_count (uniform rcb_tree tree[], uniform int node, uniform int * uniform i)
{
  if (tree[node].dimension >= 0)
  {
    leaf_count (tree, tree[node].left, i);
    leaf_count (tree, tree[node].right, i);
  }
  else (*i) ++;
}

/* O(n) split of point[] such that point[d][i<k] <= point[d][i>=k]; other dimensions are copied accordingly */
static uniform REAL quick_split (uniform int n, uniform REAL * uniform point[3], uniform int d, uniform int k)
{
  uniform REAL pivot = point[d][n/2]; /* random index might be more effective; XXX */

  uniform int d1, d2;

  switch (d)
  {
  case 0:
    d1 = 1;
    d2 = 2;
    break;
  case 1:
    d1 = 0;
    d2 = 2;
    break;
  case 2:
    d1 = 0;
    d2 = 1;
    break;
  }

  uniform int i = 0, j = n-1;

  while (i < j)
  {
    while (i < j && point[d][i] <= pivot) i++;
    while (i < j && point[d][j] > pivot) j--;

    if (i < j)
    {
      uniform REAL temp = point[d][i];
      point[d][i] = point[d][j];
      point[d][j] = temp;

      temp = point[d1][i];
      point[d1][i] = point[d1][j];
      point[d1][j] = temp;

      temp = point[d2][i];
      point[d2][i] = point[d2][j];
      point[d2][j] = temp;
    }
  }

  if (i == k) return pivot;
  else if (i > k) return quick_split (i, point, d, k);
  else
  {
    uniform REAL * uniform rpoint[3] = {point[0]+i, point[1]+i, point[2]+i};
    return quick_split (n-i, rpoint, d, k-i);
  }
}

task void rcb_tree_task (uniform int ntasks, uniform int n, uniform REAL * uniform point[3], uniform rcb_tree tree[], uniform int node)
{
  if (tree[node].dimension >= 0)
  {
    uniform REAL extents[6];

    extents_of_points (ntasks, n, point, extents);

    extents[0] = extents[3] - extents[0];
    extents[1] = extents[4] - extents[1];
    extents[2] = extents[5] - extents[2];

    uniform int dimension = 0;

    if (extents[1] > extents[0]) dimension = 1;
    if (extents[2] > extents[dimension]) dimension = 2;

    tree[node].dimension = dimension;

    uniform int left_count = 0, right_count = 0;

    leaf_count (tree, tree[node].left, &left_count);
    leaf_count (tree, tree[node].right, &right_count);

    uniform int k = (REAL) n * (REAL) left_count / (REAL) (left_count + right_count);

    tree[node].coord = quick_split (n, point, dimension, k);

    uniform REAL * uniform rpoint[3] = {point[0]+k, point[1]+k, point[2]+k};

    launch rcb_tree_task (ntasks, k, point, tree, tree[node].left);
    launch rcb_tree_task (ntasks, n-k, rpoint, tree, tree[node].right);
  }
}

/* create rcb tree; uniformly bisect untill leaf size <= cutoff; or if cutoff < 0 then create -cutoff equal size leaves */
export uniform rcb_tree * uniform rcb_tree_create (uniform int ntasks, uniform int n,
  uniform REAL * uniform point[3], uniform int cutoff, uniform int * uniform tree_size)
{
  *tree_size = 1;
  
  if (cutoff < 0)
  {
    rcb_tree_size (-cutoff, 1, tree_size);
  }
  else
  {
    cutoff = MAX (cutoff, 1); /* 0 would cause trouble */

    rcb_tree_size (n, cutoff, tree_size);
  }

  uniform rcb_tree * uniform tree = uniform new uniform rcb_tree [*tree_size];

  uniform int i = 0;

  if (cutoff < 0)
  {
    rcb_tree_init (-cutoff, 1, tree, 0, &i);
  }
  else
  {
    rcb_tree_init (n, cutoff, tree, 0, &i);
  }

  launch rcb_tree_task (ntasks, n, point, tree, 0);
  sync;

  return tree;
}

/* destroy rcb tree */
export void rcb_tree_destroy (uniform rcb_tree * uniform tree)
{
  delete tree;
}
