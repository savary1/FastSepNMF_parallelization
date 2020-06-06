# Target rules

ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif

LIBRARIES := -lm

#GCC_novec
#OPT := -O0 -fno-tree-vectorize
#GCC_autovec
#OPT := -O3 -mavx -ftree-vectorizer-verbose=6 -ffast-math
#ICC_novec
#OPT := -O0 -restrict -no-vec
#ICC_autovec
OPT := -O3 -restrict -qopt-report-phase=vec,loop,openmp -qopt-report=3

CC = icc

################################################################################

# Target rules

all: build
build: FastSEPNMF

ReadWrite.o: ReadWrite.c
	$(CC) $(OPT) -o $@ -c $< -w

FastSEPNMF.o: FastSEPNMF.c	
	$(CC) $(OPT) -o $@ -c $< -w

FastSEPNMF: FastSEPNMF.o ReadWrite.o
	$(CC) $(OPT) -o $@ $+ $(LIBRARIES)

clean:
	rm -f FastSEPNMF FastSEPNMF.o ReadWrite.o
