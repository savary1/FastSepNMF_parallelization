
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
	float norm_val = 0, aux = 0, pixel = 0;
	
	i =  blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < image_size){
        for(j = 0; j < bands; j++){
           norm_val += image_c[j*image_size + i]; 
        } 
		
		norm_val = 1.0/(norm_val + 1.0e-16);
	
		for(j = 0; j < bands; j++){
            pixel = image_c[j*image_size  + i] * norm_val;
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



void checkCUDAError(const char *mensaje, cudaError_t error){
	
	if(error != cudaSuccess){
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	
	}
}

void reservarMemoria( int bands, long int image_size, float **v_c, float **image_c, float **normM_c, float **image, float **normM, float **v){
	cudaError_t error;
	
	error = cudaMalloc(v_c, bands*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de v_c", error);
	
	error = cudaMalloc(image_c, bands*image_size*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de image_c", error);
	
	error = cudaMalloc(normM_c, image_size*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de normMc", error);
	
	error = cudaHostAlloc(image, bands*image_size*sizeof(float), cudaHostAllocDefault); 
	checkCUDAError("ERROR EN cudaHostAlloc de image", error);
	
	error = cudaHostAlloc(normM, image_size*sizeof(float), cudaHostAllocDefault);
	checkCUDAError("ERROR EN cudaHostAlloc de normM", error);
	
	error = cudaHostAlloc(v, image_size*sizeof(float), cudaHostAllocDefault);
	checkCUDAError("ERROR EN cudaHostAlloc de v", error);

}


void liberarMemoria(float *v_c, float *image_c, float *normM_c, float *image, float *normM, float *v){
	cudaError_t error;
	
	error = cudaFree(v_c);
	checkCUDAError("ERROR EN cudaFree de v_c", error);
	
	error = cudaFree(image_c);
	checkCUDAError("ERROR EN cudaFree de image_c", error);
	
	error = cudaFree(normM_c);
	checkCUDAError("ERROR EN cudaFree de normM_c", error);
	
	error = cudaFreeHost(image);
	checkCUDAError("ERROR EN cudaFreeHost de image", error);
	
	error = cudaFreeHost(normM);
	checkCUDAError("ERROR EN cudaFreeHost de normM", error);
	
	error = cudaFreeHost(v);
	checkCUDAError("ERROR EN cudaFreeHost de v", error);
	
}

void selectDevice(){
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
	checkCUDAError("ERROR EN setDevice", error);
	
	
}


void actualizarNormM(float *v, int bands, float *normM, long int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c, float *t_act){
	cudaError_t error;
	struct timeval t1, t2;
	float t_sec, t_usec;
		
	error = cudaMemcpy(v_c, v, bands*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("ERROR EN cudamemcpy de v_c", error);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 
	
	gettimeofday(&t1,NULL);
	
	actualizacion<<<dimGrid,dimBlock>>>(v_c, image_c, bands, normM_c, image_size);
	checkCUDAError("ERROR EN kernel actualización", cudaGetLastError());
	cudaDeviceSynchronize();	
	
	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	t_act[0] = t_act[0] + t_sec + t_usec/1.0e+6;
		
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudaMemcpy de normM", error);
	
	
}


void normalizeImgC(float *image, long int image_size, int bands,float *image_c, int rows, float *normM_c, float *normM, float *normM1, float *t_copia_v, float *t_normalizar, float *t_copianorm){
	cudaError_t error;
	struct timeval t1, t2;
	float t_sec, t_usec;

	gettimeofday(&t1,NULL);
	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice); // Bueno
	checkCUDAError("ERROR EN cudaMemcpy de image_c", error);
	
	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	t_copia_v[0] = t_copia_v[0] + t_sec + t_usec/1.0e+6;
		
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 

	normalizacion<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c);
	checkCUDAError("ERROR EN kernel normalización", cudaGetLastError());
	cudaDeviceSynchronize();
	
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudamemcpy de normM_c", error);


}

void calculateNormM(float *image, float *normM, float *normM1, long int image_size, int bands, int rows, float *image_c, float *normM_c){
	cudaError_t error;
	
	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("ERROR EN cudaMemcpy de image_c", error);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(ceil(image_size/1024)); 
	
	calculateNormM<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c);
	checkCUDAError("ERROR EN kernel calculateNormM", cudaGetLastError());
	cudaDeviceSynchronize();
	
	error = cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudamemcpy de normM_c", error);

}





