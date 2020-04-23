
#include "kernel.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>



__global__ void actualizacion(float *v_c, float *image_c, int bands, float *normM_c, int image_size) { 
	int k;
	float faux = 0;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	if (j < image_size){
		faux = 0;
		for(k = 0; k < bands; k++){
			faux += v_c[k] * image_c[j*bands + k];
		}
		normM_c[j] -= faux * faux;
	}

}

void check_CUDA_Error(const char *mensaje){
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	
	}
}


// Reservamos memoria para los arrays
void reservar_memoria( int bands, int image_size, float **v_c, float **image_c, float **normM_c){
	
	cudaMalloc(v_c, bands*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de v_c");
	
	cudaMalloc(image_c, bands*image_size*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de image_c");
	
	cudaMalloc(normM_c, image_size*sizeof(float));
	check_CUDA_Error("ERROR EN cudaMalloc de normMc");
	
	
}

void copiar_image(float *image ,float *image_c, int bands, int image_size){

	cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudaMemcpy de image_c");
}


void liberar_memoria(float *v_c, float *image_c, float *normM_c){

	cudaFree(v_c);
	check_CUDA_Error("ERROR EN cudaFree de v_c");
	
	cudaFree(image_c);
	check_CUDA_Error("ERROR EN cudaFree de image_c");
	
	cudaFree(normM_c);
	check_CUDA_Error("ERROR EN cudaFree de normM_c");
	
}

void select_device(){ // Devolver el numero de hilos para saber el tamaño de los bloques 
	int count;
	int	i, device;
	cudaDeviceProp prop;
	
	cudaGetDeviceCount(&count);
	
	for(i = 0; i < count; ++i){
		cudaGetDeviceProperties(&prop, i);
		
		printf("Device %d, con nombre: %s, con maximo de hilos por bloque: %d, con maximas dimensiones de cada grid: %d, con maximo hilo por bloque: %d \n", i, prop.name, prop.maxThreadsPerBlock, prop.maxGridSize[0], prop.maxThreadsDim[0]);
		
	}
	
	printf("Select a device: ");
	scanf ("%d", &device);
	
	//cudaGetDeviceProperties(prop, device);
	//hilos = prop->maxThreadsPerBlock;
	
	cudaSetDevice(device);
	check_CUDA_Error("ERROR EN setDevice");
	
}


void actualizar_normM(float *v, int bands, float *normM, int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c){

	cudaMemcpy(v_c, v, bands*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudamemcpy de v_c");
	
	//cudaMemcpy(normM_c, normM, image_size*sizeof(float), cudaMemcpyHostToDevice);
	
		
	//Hilos y bloques
	//dim3 dimBlock(image_size/rows);
	//dim3 dimGrid(rows); 
	
	//actualizacion
	actualizacion<<<dimGrid,dimBlock>>>(v_c, image_c, bands, normM_c, image_size);
	check_CUDA_Error("ERROR EN actualización");
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(normM, normM_c, image_size*sizeof(float), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudaMemcpy de normM");
	
}

void copy_normM(float *normM, float *normM_c, int image_size){

	cudaMemcpy(normM_c, normM, image_size*sizeof(float), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudamemcpy de normM_c");
}




