
CC = gcc
NVCC = nvcc

LIBRARIES := -lm 
LIBS = -L/optshared/cuda-10.0/lib64/ -lcudart -lm 

OPT := -O3 -march=native -ftree-vectorizer-verbose=6 -ffast-math -ftree-parallelize-loops=3

CFLAGS     = -O3
NVCFLAGS   = $(CFLAGS) -Wno-deprecated-gpu-targets

	
all: build
build: FastSEPNMF

ReadWrite.o: ReadWrite.c
	$(CC) $(OPT) -o $@ -c $< -w

kernel.o : kernel.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $< -I.
	
FastSEPNMF.o: FastSEPNMF.c	
	$(CC) $(OPT) -c -o $@ $< -I. 
	
FastSEPNMF: FastSEPNMF.o ReadWrite.o kernel.o
	$(CC) $(OPT)  -o $@ $? -I. $(LIBS)

clean:
	rm -f FastSEPNMF FastSEPNMF.o ReadWrite.o kernel.o


