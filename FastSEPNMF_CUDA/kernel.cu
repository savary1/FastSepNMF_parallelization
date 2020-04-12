#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernel.h"

#include <sys/time.h>
#include <sys/resource.h>


__global__ void actualizacion(float *v_c, float *image_c, int bands, float *normM_c, int image_size, float *out_c) { 
	int k = 0;
	float faux = 0;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	//int j =  blockIdx.x;
	
	if (j < image_size){
		for(k = 0; k < bands; k++){
			faux += v_c[k] * image_c[j*bands + k];
		}
		faux = faux * faux;
		//normM_c[j] -= faux ;
		out_c[j] = normM_c[j] - faux;
		
	}

}


void act(float *v, float *image, int bands, float *normM, int image_size, int i, int rows){

	float *v_c, *image_c, *normM_c, *out_c;
	
	// Reservamos memoria para los arrays
	cudaMalloc((void**)&v_c, bands*sizeof(float));
	
	cudaMalloc((void**)&image_c, bands*image_size*sizeof(float));
	cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&normM_c, image_size*sizeof(float));
	cudaMalloc((void**)&out_c, image_size*sizeof(float));
	

	cudaMemcpy(v_c, v, bands*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(normM_c, normM, image_size*sizeof(float), cudaMemcpyHostToDevice);
	
		
	//Hilos y bloques
	dim3 dimBlock(image_size/rows);
	dim3 dimGrid(rows); 
	//dim3 dimBlock(image_size);
	//dim3 dimGrid(1);
	
	//actualizacion
	actualizacion<<<dimGrid,dimBlock>>>(v_c, image_c, bands, normM_c, image_size, out_c);
	
	//cudaDeviceSynchronize();
	cudaMemcpy(normM, out_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	
	
	cudaFree(v_c);
	cudaFree(image_c);
	cudaFree(normM_c);
	cudaFree(out_c);
	

	
}






