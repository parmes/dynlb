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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "macros.h"
#include "morton_ispc.h"
#include "alloc_ispc.h"

void dynlb_morton_balance (int n, REAL *point[3], int ranks[])
{
  int size, rank, *vn, *dn, gn, i, j, k, m, *gorder, *granks;
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
    ERRMEM (gorder = aligned_int_alloc (gn));

    morton_ordering (gn, gpoint, gorder, 0);

    ERRMEM (granks = aligned_int_alloc (gn));

    m = gn / size;

    for (i = j = 0; j < size; j ++)
    {
      for (k = 0; k < m; k ++, i ++)
      {
	granks[gorder[i]] = j;
      }
    }

    m = gn % size;

    for (k = 0; k < m; k++, i ++) /* remainder m is distributed to first m ranks */
    {
      granks[gorder[i]] = k;
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
    aligned_real_free (gpoint[0]);
    aligned_real_free (gpoint[1]);
    aligned_real_free (gpoint[2]);
    free (dn);
    free (vn);
  }
}

int main (int argc, char *argv[])
{
  int m, n, i, *ranks, rank;
  REAL *point[3];

  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  srand(time(NULL) + rank);

  if (argc == 2) m = atoi(argv[1]);
  else
  {
    printf ("SYNOPSIS: dynlb max_domain_size\n");
    return 1;
  }

  n = rand() % m + 1;

  ERRMEM (point[0] = malloc (n * sizeof (REAL)));
  ERRMEM (point[1] = malloc (n * sizeof (REAL)));
  ERRMEM (point[2] = malloc (n * sizeof (REAL)));
  ERRMEM (ranks = malloc (n * sizeof (int)));

  for (i = 0; i < n; i ++)
  {
    point[0][i] = DRAND();
    point[1][i] = DRAND();
    point[2][i] = DRAND();
  }

  dynlb_morton_balance (n, point, ranks);

  printf ("rank %d size %d export ranks: ", rank, n);

  for (i = 0; i < n; i ++)
  {
    printf ("%d ", ranks[i]);
  }

  printf ("\n");

  free (point[0]);
  free (point[1]);
  free (point[2]);
  free (ranks);

  MPI_Finalize ();

  return 0;
}
