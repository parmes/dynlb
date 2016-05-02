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
#include "rcb.h"

task void dummy (uniform int n, uniform int * uniform i)
{
  if (n > 1)
  {
    launch dummy (n/2, i);
    launch dummy (n-n/2, i);
  }
  else
  {
    atomic_add_global (i, 1);
    print ("leaf no. %\n", *i);
  }
}

/* create rcb tree */
export uniform rcb_tree * uniform rcb_tree_create (uniform int ntasks, uniform int n,
  uniform REAL * uniform point[3], uniform int cutoff, uniform int * uniform tree_size)
{
  return NULL;
}

/* destroy rcb tree */
export void rcb_tree_destroy (uniform rcb_tree * uniform rcbtree)
{
  delete rcbtree;
}
