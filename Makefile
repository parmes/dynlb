include Config.mak

# C files
C_SRC=dynlb.c

# C++ files
CPP_SRC=tasksys.cpp

# ISPC files
ISPC_SRC=alloc.ispc sort.ispc morton.ispc radix.ispc rcb.ispc part.ispc simu.ispc

# ISPC targets
ISPC_TARGETS=sse2,sse4,avx

# Library name
LIB=libdynlb

# Program name
EXE=test_dynlb

# Do the rest
include common.mk
