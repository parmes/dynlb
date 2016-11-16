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
#include "sort.h"

typedef unsigned int uint;

/* calculate extrema of x, y, z */
task void _dynlb_extrema (uniform int span, uniform int n, uniform REAL x[], uniform REAL y[], uniform REAL z[], uniform REAL extents[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  REAL e[6] = {REAL_MAX,REAL_MAX,REAL_MAX,-REAL_MAX,-REAL_MAX,-REAL_MAX};

  foreach (i = start ... end)
  {
    if (x[i] < e[0]) e[0] = x[i];
    if (y[i] < e[1]) e[1] = y[i];
    if (z[i] < e[2]) e[2] = z[i];
    if (x[i] > e[3]) e[3] = x[i];
    if (y[i] > e[4]) e[4] = y[i];
    if (z[i] > e[5]) e[5] = z[i];
  }

  uniform REAL * uniform out = &extents [6*taskIndex];

  out[0] = reduce_min (e[0]);
  out[1] = reduce_min (e[1]);
  out[2] = reduce_min (e[2]);
  out[3] = reduce_max (e[3]);
  out[4] = reduce_max (e[4]);
  out[5] = reduce_max (e[5]);
}

/* Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit */
/* https://developer.nvidia.com/content/thinking-parallel-part-iii-tree-construction-gpu */
inline uint expandbits(uint v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

/* Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1] */
/* https://developer.nvidia.com/content/thinking-parallel-part-iii-tree-construction-gpu */
task void _dynlb_morton (uniform int span, uniform int n, uniform REAL x[], uniform REAL y[], uniform REAL z[], uniform REAL extents[], uniform uint code[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  uniform REAL wx = extents[3]-extents[0],
               wy = extents[4]-extents[1],
	       wz = extents[5]-extents[2];

  foreach (i = start ... end)
  {
    REAL px = (x[i]-extents[0])/wx,
         py = (y[i]-extents[1])/wy,
	 pz = (z[i]-extents[2])/wz;

    REAL qx = min(max(px * 1024.0f, 0.0f), 1023.0f),
         qy = min(max(py * 1024.0f, 0.0f), 1023.0f),
         qz = min(max(pz * 1024.0f, 0.0f), 1023.0f);

    uint xx = expandbits((int)qx),
         yy = expandbits((int)qy),
         zz = expandbits((int)qz);

    code[i] = xx*4 + yy*2 + zz;
  }
}

/* morton ordering */
export void _dynlb_morton_ordering (uniform int ntasks, uniform int n, uniform REAL * uniform point[3], uniform unsigned int code[], uniform int order[])
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;

  uniform REAL * uniform extents = uniform new uniform REAL [6*num];

  launch[num] _dynlb_extrema (span, n, point[0], point[1], point[2], extents);
  sync;

  for (uniform int i = 1; i < num; i ++)
  {
    uniform REAL * uniform e = &extents [6*i];

    if (e[0] < extents[0]) extents[0] = e[0];
    if (e[1] < extents[1]) extents[1] = e[1];
    if (e[2] < extents[2]) extents[2] = e[2];
    if (e[3] > extents[3]) extents[3] = e[3];
    if (e[4] > extents[4]) extents[4] = e[4];
    if (e[5] > extents[5]) extents[5] = e[5];
  }

  launch[num] _dynlb_morton (span, n, point[0], point[1], point[2], extents, code);
  sync;

  foreach (k = 0 ... n) order[k] = k;

  if (n < 10000) quick_sort (n, code, order);
  else radix_sort (num, n, code, order);

  delete extents;
}

/* task based and vectorized extents of points */
void extents_of_points (uniform int ntasks, uniform int n, uniform REAL * uniform point[3], uniform REAL extents[])
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;

  uniform REAL * uniform task_extents = uniform new uniform REAL [6*num];

  launch[num] _dynlb_extrema (span, n, point[0], point[1], point[2], task_extents);
  sync;

  extents[0] = task_extents[0];
  extents[1] = task_extents[1];
  extents[2] = task_extents[2];
  extents[3] = task_extents[3];
  extents[4] = task_extents[4];
  extents[5] = task_extents[5];

  for (uniform int i = 1; i < num; i ++)
  {
    uniform REAL * uniform e = &task_extents [6*i];

    if (e[0] < extents[0]) extents[0] = e[0];
    if (e[1] < extents[1]) extents[1] = e[1];
    if (e[2] < extents[2]) extents[2] = e[2];
    if (e[3] > extents[3]) extents[3] = e[3];
    if (e[4] > extents[4]) extents[4] = e[4];
    if (e[5] > extents[5]) extents[5] = e[5];
  }

  delete task_extents;
}
