import shutil

def inplace_change(filename, pairs):
  # Safely read the input filename using 'with'
  with open(filename) as f:
    s = f.read()

  # Safely write the changed content, if found in the file
  with open(filename, 'w') as f:
    for p in pairs:
      s = s.replace(p[0], p[1])
    f.write(s)

print 'Generating dynlb4.h, dynlb8.h ...',

shutil.copyfile ('dynlb.h', 'dynlb4.h')
inplace_change ('dynlb4.h', [('REAL', 'float')])
shutil.copyfile ('dynlb.h', 'dynlb8.h')
inplace_change ('dynlb8.h', [('REAL', 'double')])

print 'done.'
