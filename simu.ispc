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

task void step_task (uniform int span, uniform int n, uniform REAL * uniform point[3], uniform REAL * uniform velo[3], uniform REAL step)
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    point [0][i] += velo[0][i] * step;
    point [1][i] += velo[1][i] * step;
    point [2][i] += velo[2][i] * step;
  }

  foreach (i = start ... end)
  {
    if (point [0][i] < 0.0 || point[0][i] > 1.0) velo[0][i] *= -1;
    if (point [1][i] < 0.0 || point[1][i] > 1.0) velo[0][i] *= -1;
    if (point [2][i] < 0.0 || point[2][i] > 1.0) velo[0][i] *= -1;
  }
}

export void unit_cube_step (uniform int ntasks, uniform int n, uniform REAL * uniform point[3], uniform REAL * uniform velo[3], uniform REAL step)
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;


  launch[num] step_task (span, n, point, velo, step);
  sync;
}
