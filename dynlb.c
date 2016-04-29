/*
The MIT License (MIT)

Copyright (c) 2016 Tomasz Koziara

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include "macros.h"
#include "morton_ispc.h"
#include "alloc_ispc.h"
#include "part_ispc.h"
#include "dynlb.h"

/* simple morton ordering based point balancer */
void dynlb_morton_balance (int n, REAL *point[3], int ranks[])
{
  int size, rank, *vn, *dn, gn, i, j, k, m, r, *gorder, *granks;
  unsigned int *gcode;
  REAL *gpoint[3];

  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  if (rank == 0)
  {
    ERRMEM (vn = malloc (size * sizeof(int)));
  }
  else
  {
    vn = NULL;
  }

  MPI_Gather (&n, 1, MPI_INT, vn, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {

    ERRMEM (dn = malloc (size * sizeof(int)));

    for (gn = i = 0; i < size; i ++)
    {
      dn[i] = gn;
      gn += vn[i];
    }

    ERRMEM (gpoint[0] = aligned_real_alloc (gn));
    ERRMEM (gpoint[1] = aligned_real_alloc (gn));
    ERRMEM (gpoint[2] = aligned_real_alloc (gn));
  }

  MPI_Gatherv (point[0], n, MPI_INT, gpoint[0], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv (point[1], n, MPI_INT, gpoint[1], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv (point[2], n, MPI_INT, gpoint[2], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    ERRMEM (gcode = aligned_uint_alloc (gn));

    ERRMEM (gorder = aligned_int_alloc (gn));

    morton_ordering (0, gn, gpoint, gcode, gorder);

    ERRMEM (granks = aligned_int_alloc (gn));

    m = gn / size;

    r = gn % size;

    for (i = j = 0; j < size; j ++)
    {
      for (k = 0; k < m + (r ? 1 : 0); k ++, i ++) /* remainder r is distributed to first r ranks */
      {
	granks[gorder[i]] = j;
      }

      if (r) r --; /* decrement remainder */
    }

#if DEBUG
   printf ("gn = %d\n", gn);

   printf ("gorder = ");
   for (i = 0; i < gn; i ++)
   {
     printf ("%d ", gorder[i]);
   }
   printf ("\n");

   printf ("granks = ");
   for (i = 0; i < gn; i ++)
   {
     printf ("%d ", granks[i]);
   }
   printf ("\n");

   int *nn = calloc (size, sizeof(int));
   for (i = 0; i < gn; i ++) nn[granks[i]] ++;
   printf ("new counts = ");
   for (i = j = 0; i < size; i ++)
   {
     printf ("%d ", nn[i]);
     j += nn[i];
   }
   printf ("\n");
   free (nn);

   printf ("sumed up new counts = %d\n", j);
#endif
  }

  MPI_Scatterv (granks, vn, dn, MPI_INT, ranks, n, MPI_INT, 0, MPI_COMM_WORLD);

#if DEBUG
  for (i = 0; i < n; i ++)
  {
    if (ranks[i] < 0 || ranks[i] >= size)
    {
      ASSERT (0, "Export rank out of bounds");
    }
  }
#endif

  if (rank == 0)
  {
    aligned_int_free (granks);
    aligned_int_free (gorder);
    aligned_uint_free (gcode);
    aligned_real_free (gpoint[0]);
    aligned_real_free (gpoint[1]);
    aligned_real_free (gpoint[2]);
    free (dn);
    free (vn);
  }
}

/* create balancer */
struct dynlb* dynlb_create (int ntasks, int n, REAL *point[3], int cutoff, REAL epsilon)
{
  int size, rank, *vn, *dn, gn, i, *rank_size;
  struct partitioning *ptree;
  struct dynlb *lb;
  REAL *gpoint[3];
  int leaf_count;

  ERRMEM (lb = malloc (sizeof(struct dynlb)));
  lb->ntasks = ntasks;
  lb->cutoff = cutoff;
  lb->epsilon = epsilon;

  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  if (rank == 0)
  {
    ERRMEM (vn = malloc (size * sizeof(int)));
  }
  else
  {
    vn = NULL;
  }

  MPI_Gather (&n, 1, MPI_INT, vn, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {

    ERRMEM (dn = malloc (size * sizeof(int)));

    for (gn = i = 0; i < size; i ++)
    {
      dn[i] = gn;
      gn += vn[i];
    }

    ERRMEM (gpoint[0] = aligned_real_alloc (gn));
    ERRMEM (gpoint[1] = aligned_real_alloc (gn));
    ERRMEM (gpoint[2] = aligned_real_alloc (gn));
  }

  MPI_Gatherv (point[0], n, MPI_INT, gpoint[0], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv (point[1], n, MPI_INT, gpoint[1], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv (point[2], n, MPI_INT, gpoint[2], vn, dn, MPI_INT, 0, MPI_COMM_WORLD);

  ERRMEM (rank_size = calloc (size, sizeof (int)));

  if (rank == 0)
  {
    ptree = partitioning_create (ntasks, gn, gpoint, cutoff, &lb->ptree_size, &leaf_count);

    partitioning_assign_ranks (ptree, leaf_count / size, leaf_count % size);

    partitioning_store (ntasks, ptree, gn, gpoint);

#if DEBUG
    printf ("leaf count = %d\n", leaf_count);

    printf ("leaf ranks: ");
    for (i = 0; i < lb->ptree_size; i ++)
    {
      if (ptree[i].dimension < 0) /* leaf */
      {
	printf ("%d ", ptree[i].rank);
      }
    }
    printf ("\n");

    printf ("leaf sizes: ");
    for (i = 0; i < lb->ptree_size; i ++)
    {
      if (ptree[i].dimension < 0) /* leaf */
      {
	printf ("%d ", ptree[i].size);
      }
    }
    printf ("\n");
#endif

    /* determine initial balance */

    for (i = 0; i < lb->ptree_size; i ++)
    {
      if (ptree[i].dimension < 0) /* leaf */
      {
	rank_size[ptree[i].rank] += ptree[i].size;
      }
    }

    int min_size = INT_MAX, max_size = 0;

    for (i = 0; i < size; i ++)
    {
      min_size = MIN (min_size, rank_size[i]);
      max_size = MAX (max_size, rank_size[i]);
    }

    lb->initial = (REAL)max_size/(REAL)min_size;

    lb->imbalance = lb->initial;

  }

  /* broadcast ptree_size and initial imbalance */
  MPI_Bcast (lb, sizeof(struct dynlb), MPI_BYTE, 0, MPI_COMM_WORLD);

  /* broadcast rank_size and update lb->npoint */
  MPI_Bcast (rank_size, size * sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);

  lb->npoint = rank_size[rank];

  if (rank == 0)
  {
    lb->ptree = ptree;
  }
  else
  {
    lb->ptree = partitioning_alloc (lb->ptree_size);
  }

  /* broadcast ptree */
  MPI_Bcast (lb->ptree, lb->ptree_size*sizeof(struct partitioning), MPI_BYTE, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    aligned_real_free (gpoint[0]);
    aligned_real_free (gpoint[1]);
    aligned_real_free (gpoint[2]);
    free (dn);
    free (vn);
  }

  free (rank_size);

  return lb;
}

/* assign MPI rank to a point; return rank */
int dynlb_point_assign (struct dynlb *lb, REAL point[])
{
  return partitioning_point_assign (lb->ptree, 0, point);
}

/* assign MPI ranks to a box spanned between lo and hi points; return number of ranks */
int dynlb_box_assign (struct dynlb *lb, REAL lo[], REAL hi[], int ranks[])
{
  int count = 0;

  partitioning_box_assign (lb->ptree, 0, lo, hi, ranks, &count);

  return count;
}

/* update balancer */
void dynlb_update (struct dynlb *lb, int n, REAL *point[3])
{
  int i, rank, size, *local_size, *rank_size;
  struct partitioning *ptree = lb->ptree;

  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  partitioning_store (lb->ntasks, ptree, n, point);

  ERRMEM (local_size = calloc (size, sizeof (int)));

  ERRMEM (rank_size = calloc (size, sizeof (int)));

  for (i = 0; i < lb->ptree_size; i ++)
  {
    if (ptree[i].dimension < 0) /* leaf */
    {
      local_size[ptree[i].rank] += ptree[i].size;
    }
  }

  /* reduce global sizes per rank */
  MPI_Allreduce (local_size, rank_size, size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  lb->npoint = rank_size[rank];

  int min_size = INT_MAX, max_size = 0;

  for (i = 0; i < size; i ++)
  {
    min_size = MIN (min_size, rank_size[i]);
    max_size = MAX (max_size, rank_size[i]);
  }

  lb->imbalance = (REAL)max_size/(REAL)min_size;

  free (rank_size);

  if (lb->imbalance > lb->initial+lb->epsilon) /* update partitioning */
  {
    struct dynlb *dy = dynlb_create (lb->ntasks, n, point, lb->cutoff, lb->epsilon);

    partitioning_destroy (lb->ptree);
    lb->ptree = dy->ptree;
    lb->ptree_size = dy->ptree_size;

    lb->initial = dy->initial;
    lb->imbalance = dy->imbalance;
    lb->npoint = dy->npoint;

    free (dy);
  }
}

/* destroy balancer */
void dynlb_destroy (struct dynlb *lb)
{
  partitioning_destroy (lb->ptree);
  free (lb);
}
