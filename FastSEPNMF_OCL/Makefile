# Target rules

# ifdef OPENCL_INC
  CL_CFLAGS = -I/optshared/cuda/include/CL
# endif

# ifdef OPENCL_LIB
  CL_LDFLAGS = -L/optshared/cuda/lib64
# endif

LIBRARIES := -lOpenCL -lm 

#GCC_novec
#OPT := -O1 -fno-tree-vectorize -ffast-math 
#GCC_autovec
#OPT := -O3 -mavx -ftree-vectorizer-verbose=6 -ffast-math -ftree-parallelize-loops=3
#ICC_novec
#OPT := -O0 -restrict -no-vec
#ICC_autovec
OPT := -O3 -restrict -qopt-report-phase=vec,loop,openmp -qopt-report=3 -qopenmp

CC = icc

################################################################################

# Target rules

all: build
build: FastSEPNMF

ReadWrite.o: ReadWrite.c
	$(CC) $(OPT) $(CL_LDFLAGS) -o $@ -c $< -w -g

FastSEPNMF.o: FastSEPNMF.c	
	$(CC) $(OPT) $(CL_LDFLAGS) -o $@ -c $< -w -g

FastSEPNMF: FastSEPNMF.o ReadWrite.o
	$(CC) $(OPT) $(CL_LDFLAGS) -o $@ $+ $(LIBRARIES) -g

clean:
	rm -f FastSEPNMF FastSEPNMF.o ReadWrite.o
