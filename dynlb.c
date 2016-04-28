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

struct dynlb /* load balancer interface */
{
  int ntasks;
  int cutoff;
  struct partitioning *ptree;
  int ptree_size;
  int leaf_count;
};

/* create balancer */
dynlb dynlb_create (int ntasks, int n, REAL *point[3])
{
  int size, rank, *vn, *dn, gn, i;
  struct partitioning *ptree;
  REAL *gpoint[3];
  int tree_size;
  int leaf_count;
  int cutoff;

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
    cutoff = (gn / size) / 8; /* TODO */

    ptree = partitioning_create (0, gn, gpoint, cutoff, &tree_size, &leaf_count);

    partitioning_assign_ranks (ptree, size, leaf_count / size, leaf_count % size);

    partitioning_store (0, ptree, gn, gpoint);

    /* TODO: determine initial balance */
  }

  /* MPI_Scatter (granks, vn, dn, MPI_INT, ranks, n, MPI_INT, 0, MPI_COMM_WORLD); */

  if (rank == 0)
  {
    aligned_real_free (gpoint[0]);
    aligned_real_free (gpoint[1]);
    aligned_real_free (gpoint[2]);
    free (dn);
    free (vn);
  }

  return NULL;
}

/* assign MPI rank to a point; return rank */
int dynlb_point_assign (dynlb lb, REAL point[])
{
  return 0;
}

/* assign MPI ranks to a box spanned between lo and hi points; return number of ranks */
int dynlb_box_assign (dynlb lb, REAL lo[], REAL hi[], int ranks[])
{
  return 0;
}

/* update balancer */
void dynlb_update (dynlb lb, int n, REAL *point[3])
{
}

/* destroy balancer */
void dynlb_destroy (dynlb lb)
{
}
