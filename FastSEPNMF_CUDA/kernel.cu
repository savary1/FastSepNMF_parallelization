
#include "kernel.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>


__global__ void actualizacion(float *v_c, float *image_c, int bands, float *normM_c, long int image_size) { 
	int k;
	float faux = 0;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	if (j < image_size){
		faux = 0;
		for(k = 0; k < bands; k++){
			faux += v_c[k] * image_c[k*image_size + j];
		}
		normM_c[j] -= faux * faux;
	}

}


__global__ void normalizacion(float *image_c, int bands, long int image_size, float *normM_c) { 
	long int j, i;
	float normVal = 0, aux = 0, pixel = 0;
	
	i =  blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < image_size){
        for(j = 0; j < bands; j++){
           normVal += image_c[j*image_size + i]; 
        } 
		
		normVal = 1.0/(normVal + 1.0e-16);
	
		for(j = 0; j < bands; j++){
            pixel = image_c[j*image_size  + i] * normVal;
            image_c[j*image_size + i] = pixel;
            aux += pixel * pixel;
        }
        normM_c[i] = aux;
	}

}

__global__ void calculateNormM(float *image_c, int bands, long int image_size, float *normM_c) { 
	int k;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < image_size){
		for(k = 0; k < bands; k++){
				normM_c[j] += image_c[k*image_size + j] * image_c[k*image_size + j]; 
			}
	}

}



void check_CUDA_Error(const char *mensaje, cudaError_t error){
	//cudaDeviceSynchronize();
	if(error != cudaSuccess){
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	
	}
}

// Reservamos memoria para los arrays
void reservar_memoria( int bands, long int image_size, float **v_c, float **image_c, float **normM_c){
	cudaError_t error;
	
	error = cudaMalloc(v_c, bands*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de v_c", error);
	
	error = cudaMalloc(image_c, bands*image_size*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de image_c", error);
	
	error = cudaMalloc(normM_c, image_size*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de normMc", error);
	
}


void liberar_memoria(float *v_c, float *image_c, float *normM_c){
	cudaError_t error;
	
	error = cudaFree(v_c);
	check_CUDA_Error("ERROR EN cudaFree de v_c", error);
	
	error = cudaFree(image_c);
	check_CUDA_Error("ERROR EN cudaFree de image_c", error);
	
	error = cudaFree(normM_c);
	check_CUDA_Error("ERROR EN cudaFree de normM_c", error);
	
}

void select_device(){
	int count;
	int	i, device;
	cudaDeviceProp prop;
	cudaError_t error;
	
	cudaGetDeviceCount(&count);
	
	for(i = 0; i < count; ++i){
		cudaGetDeviceProperties(&prop, i);
		
		printf("Device %d, con nombre: %s\n", i, prop.name);
		
	}
	
	printf("Select a device: ");
	scanf ("%d", &device);
		
	error = cudaSetDevice(device);
	check_CUDA_Error("ERROR EN setDevice", error);
	
}


void actualizar_normM(float *v, int bands, float *normM, long int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c){
	cudaError_t error;

	error = cudaMemcpy(v_c, v, bands*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudamemcpy de v_c", error);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 
	
	//dim3 dimBlock(1);
	//dim3 dimGrid(ceil(image_size)); 
	
	//dim3 dimBlock(image_size/rows);
	//dim3 dimGrid(rows); 
	
	//actualizacion
	actualizacion<<<dimGrid,dimBlock>>>(v_c, image_c, bands, normM_c, image_size);
	cudaDeviceSynchronize();	
		
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudaMemcpy de normM", error);
	
}


void normalize_imgC(float *image, long int image_size, int bands,float *image_c, int rows, float *normM_c, float *normM, float *normM1){
	cudaError_t error;
	
	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudaMemcpy de image_c", error);

	
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 
	
	//dim3 dimBlock(1);
	//dim3 dimGrid(ceil(image_size)); 
	
	//dim3 dimBlock(image_size/rows);
	//dim3 dimGrid(rows); 
	
	//normalizacion
	normalizacion<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c);
	cudaDeviceSynchronize();
	
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c", error);
	
	error = cudaMemcpy(normM1, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c", error);

}

void calculate_normM(float *image, float *normM, float *normM1, long int image_size, int bands, int rows, float *image_c, float *normM_c){
	cudaError_t error;
	
	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudaMemcpy de image_c", error);

	
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 
	
	//dim3 dimBlock(1);
	//dim3 dimGrid(ceil(image_size)); 
	
	//dim3 dimBlock(image_size/rows);
	//dim3 dimGrid(rows);  
	
	calculateNormM<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c);
	cudaDeviceSynchronize();
	
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c", error);
	
	error = cudaMemcpy(normM1, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c", error);
}


void copy_normM(float *normM, float *normM_c, long int image_size){
	cudaError_t error;
	
	error = cudaMemcpy(normM_c, normM, image_size*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c", error);
}




