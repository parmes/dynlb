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

task void histogram (uniform int span, uniform int n, uniform int64 code[], uniform int pass, uniform int hist[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;
  uniform int strip = (end-start)/programCount;
  uniform int tail = (end-start)%programCount;
  int i = programCount*taskIndex + programIndex;
  int g [256];

  cfor (int j = 0; j < 256; j ++)
  {
    g[j] = 0;
  }

  cfor (int k = start+programIndex*strip; k < start+(programIndex+1)*strip; k ++)
  {
    unsigned int8 *c = (unsigned int8*) &code[k];

    g[c[pass]] ++;
  }

  if (programIndex == programCount-1) /* remainder is processed by the last lane */
  {
    for (int k = start+programCount*strip; k < start+programCount*strip+tail; k ++)
    {
      unsigned int8 *c = (unsigned int8*) &code[k];

      g[c[pass]] ++;
    }
  }

  cfor (int j = 0; j < 256; j ++)
  {
    hist[j*programCount*taskCount+i] = g[j];
  }
}

task void permutation (uniform int span, uniform int n, uniform int64 code[], uniform int pass, uniform int hist[], uniform int64 perm[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;
  uniform int strip = (end-start)/programCount;
  uniform int tail = (end-start)%programCount;
  int i = programCount*taskIndex + programIndex;
  int g [256];

  cfor (int j = 0; j < 256; j ++)
  {
    g[j] = hist[j*programCount*taskCount+i];
  }

  cfor (int k = start+programIndex*strip; k < start+(programIndex+1)*strip; k ++)
  {
    unsigned int8 *c = (unsigned int8*) &code[k];

    int l = g[c[pass]];

    perm[l] = code[k];

    g[c[pass]] = l+1;
  }

  if (programIndex == programCount-1) /* remainder is processed by the last lane */
  {
    for (int k = start+programCount*strip; k < start+programCount*strip+tail; k ++)
    {
      unsigned int8 *c = (unsigned int8*) &code[k];

      int l = g[c[pass]];

      perm[l] = code[k];

      g[c[pass]] = l+1;
    }
  }
}

task void copy (uniform int span, uniform int n, uniform int64 from[], uniform int64 to[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    to[i] = from[i];
  }
}

task void pack (uniform int span, uniform int n, uniform unsigned int code[], uniform int64 pair[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    pair[i] = ((int64)i<<32)+code[i];
  }
}

task void unpack (uniform int span, uniform int n, uniform int64 pair[], uniform int unsigned code[], uniform int order[])
{
  uniform int start = taskIndex*span;
  uniform int end = taskIndex == taskCount-1 ? n : start+span;

  foreach (i = start ... end)
  {
    code[i] = pair[i];
    order[i] = pair[i]>>32;
  }
}

task void addup (uniform int h[], uniform int g[])
{
  uniform int * uniform u = &h[256*programCount*taskIndex];
  uniform int i, x, y = 0;

  for (i = 0; i < 256*programCount; i ++)
  {
    x = u[i];
    u[i] = y;
    y += x;
  }

  g[taskIndex] = y;
}

task void bumpup (uniform int h[], uniform int g[])
{
  uniform int * uniform u = &h[256*programCount*taskIndex];
  uniform int z = g[taskIndex];

  foreach (i = 0 ... 256*programCount)
  {
    u[i] += z;
  }
}

static void prefix_sum (uniform int num, uniform int h[])
{
  uniform int * uniform g = uniform new uniform int [num+1];
  uniform int i;

  launch[num] addup (h, g+1);
  sync;

  for (g[0] = 0, i = 1; i < num; i ++) g[i] += g[i-1];

  launch[num] bumpup (h, g);
  sync;

  delete g;
}

/* parallel radix sort on unsigned integers */
void radix_sort (uniform int n, uniform unsigned int code[], uniform int order[], uniform int ntasks)
{
  uniform int num = ntasks < 1 ? num_cores () : ntasks;
  uniform int span = n / num;
  uniform int hsize = 256*programCount*num;
  uniform int * uniform hist = uniform new uniform int [hsize];
  uniform int64 * uniform pair = uniform new uniform int64 [n];
  uniform int64 * uniform temp = uniform new uniform int64 [n];
  uniform int pass, i;

#if DEBUG
  if (n < 100)
  {
    print ("input: ");
    for (i = 0; i < n; i ++) print ("%, ", code[i]);
    print ("\n");
  }
#endif

  launch[num] pack (span, n, code, pair);
  sync;

  for (pass = 0; pass < 4; pass ++)
  {
    launch[num] histogram (span, n, pair, pass, hist);
    sync;

    prefix_sum (num, hist);

    launch[num] permutation (span, n, pair, pass, hist, temp);
    sync;

    launch[num] copy (span, n, temp, pair);
    sync;
  }

  launch[num] unpack (span, n, pair, code, order);
  sync;

#if DEBUG
  for (i = 0; i < n; i ++)
  {
    if (i > 0 && code[i-1] > code[i])
      print ("ERR at % => % > %; ", i, code[i-1], code[i]);
  }

  if (n < 100)
  {
    print ("output: ");
    for (i = 0; i < n; i ++) print ("%, ", code[i]);
    print ("\n");
    print ("order: ");
    for (i = 0; i < n; i ++) print ("%, ", order[i]);
    print ("\n");
  }
#endif

  delete hist;
  delete pair;
  delete temp;
}

/* serial quick sort on unsigned integers */
void quick_sort (uniform int n, uniform unsigned int a[], uniform int order[])
{
  uniform unsigned int p, t;
  uniform int i, j;

  if (n < 2) return;

  p = a[n/2];

  for (i = 0, j = n - 1;; i++, j--)
  {
    while (a[i] < p) i++;
    while (p < a[j]) j--;
    if (i >= j) break;

    t = a[i];
    a[i] = a[j];
    a[j] = t;

    t = order[i];
    order[i] = order[j];
    order[j] = t;
  }

  quick_sort (i, a, order);
  quick_sort (n-i, a+i, order+i);
}
