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
#include "dynlb.h"

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

#if 0
  dynlb_morton_balance (n, point, ranks);

  printf ("rank %d size %d export ranks: ", rank, n);

  for (i = 0; i < n; i ++)
  {
    printf ("%d ", ranks[i]);
  }

  printf ("\n");
#else
  printf ("rank %d n = %d\n", rank, n);

  struct dynlb *lb = dynlb_create (0, n, point, 4, 0.5);

  if (rank == 0) printf ("dynlb initial balance = %g\n", lb->initial);

  printf ("rank %d npoint = %d\n", rank, lb->npoint);
#endif

  free (point[0]);
  free (point[1]);
  free (point[2]);
  free (ranks);

  MPI_Finalize ();

  return 0;
}
