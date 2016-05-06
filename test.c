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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "macros.h"
#include "dynlb.h"
#include "timer.h"
#include "simu_ispc.h"

double ptimerend (struct timing *t)
{
  double local, time;

  local = timerend (t);

  MPI_Allreduce (&local, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return time;
}

int main (int argc, char *argv[])
{
  int max_points_per_rank = 100;
  int num_time_steps = 100;
  REAL time_step = 0.001;
  int n, i, rank, size, *ranks;
  REAL *point[3], *velo[3];
  struct timing t;
  double dt[2], gt[2];

  if (argc == 1)
  {
    printf ("SYNOPSIS: dynlb max_points_per_rank [num_time_steps = 100] [time_step = 0.001]\n");
    return 0;
  }

  if (argc >= 2) max_points_per_rank = atoi(argv[1]);

  if (argc >= 3) num_time_steps = atoi(argv[2]);

  if (argc >= 4) time_step = atof(argv[3]);

  MPI_Init (&argc, &argv);

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  MPI_Comm_size (MPI_COMM_WORLD, &size);

  srand(time(NULL) + rank);

  n = rand() % max_points_per_rank + 1;

  if (rank == 0) printf ("Generating random points in unit cube...\n");

  timerstart (&t);

  printf ("Generating %d points on rank %d.\n", n, rank);

  ERRMEM (point[0] = malloc (n * sizeof (REAL)));
  ERRMEM (point[1] = malloc (n * sizeof (REAL)));
  ERRMEM (point[2] = malloc (n * sizeof (REAL)));
  ERRMEM (velo[0] = malloc (n * sizeof (REAL)));
  ERRMEM (velo[1] = malloc (n * sizeof (REAL)));
  ERRMEM (velo[2] = malloc (n * sizeof (REAL)));
  ERRMEM (ranks = malloc (n * sizeof (int)));

  for (i = 0; i < n; i ++)
  {
    point[0][i] = DRAND();
    point[1][i] = DRAND();
    point[2][i] = DRAND();
    velo[0][i] = DRAND()-0.5;
    velo[1][i] = DRAND()-0.5;
    velo[2][i] = DRAND()-0.5;
  }

  dt[0] = ptimerend (&t);

  if (rank == 0) printf ("Generating points took %g sec.\n", dt[0]);

  if (rank == 0) printf ("Timing %d simple morton based balancing steps...\n", num_time_steps);

  for (i = 0, dt[0] = 0.0, dt[1] = 0.0; i < num_time_steps; i ++)
  {
    timerstart (&t);

    dynlb_morton_balance (n, point, ranks);

    dt[0] += timerend (&t);

    if (rank == 0) printf ("."), fflush (stdout);

    timerstart (&t);

    unit_cube_step (0, n, point, velo, time_step);

    dt[1] += timerend (&t);
  }

  MPI_Allreduce (dt, gt, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  gt[0] /= (double)size * (double)num_time_steps;
  gt[1] /= (double)size * (double)num_time_steps;

  if (rank == 0) printf ("\nMORTON: avg. integration: %g sec. per step, avg. balancing: %g sec. per step; ratio: %g\n", gt[0]+gt[1], gt[1], gt[1]/(gt[0]+gt[1]));

  if (rank == 0) printf ("Creating partitioning tree based balancer ...\n");

  timerstart (&t);

  struct dynlb *lb = dynlb_create (0, n, point, 0, 0.5, DYNLB_RCB_TREE);

  dt[0] += ptimerend (&t);

  if (rank == 0) printf ("Took %g sec.\nInitial imbalance %g\n", dt[0], lb->initial);

  if (rank == 0) printf ("Timing %d partitioning tree based balancing steps...\n", num_time_steps);

  for (i = 0, dt[0] = 0.0, dt[1] = 0.0; i < num_time_steps; i ++)
  {
    timerstart (&t);

    dynlb_update (lb, n, point);

    dt[0] += timerend (&t);

    if (rank == 0) printf ("Step %d imbalance %g\n", i, lb->imbalance);

    timerstart (&t);

    unit_cube_step (0, n, point, velo, time_step);

    dt[1] += timerend (&t);
  }

  MPI_Allreduce (dt, gt, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  gt[0] /= (double)size * (double)num_time_steps;
  gt[1] /= (double)size * (double)num_time_steps;

  if (rank == 0) printf ("TREE: avg. integration: %g sec. per step, avg. balancing: %g sec. per step; ratio: %g\n", gt[0]+gt[1], gt[1], gt[1]/(gt[0]+gt[1]));

  free (point[0]);
  free (point[1]);
  free (point[2]);
  free (velo[0]);
  free (velo[1]);
  free (velo[2]);
  free (ranks);

  MPI_Finalize ();

  return 0;
}
