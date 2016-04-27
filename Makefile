# C compiler
CC=mpicc

# C++ compiler
CXX=mpicxx

# C files
C_SRC=dynlb.c

# C++ files
CPP_SRC=tasksys.cpp

# ISPC files
ISPC_SRC=alloc.ispc sort.ispc morton.ispc

# ISPC targets
ISPC_TARGETS=sse2,sse4,avx

# Program name
EXE=dynlb

# Floating point type
REAL=float

# Debug version
DEBUG=yes

# Do the rest
include common.mk