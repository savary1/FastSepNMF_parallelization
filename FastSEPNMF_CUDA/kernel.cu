
#include "kernel.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <sys/time.h>
#include <sys/resource.h>

#define DIM 224

__global__ void maxVal(float *normM_c, long int image_size, float *d_projections){
	__shared__ float val[1024];
	
	unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x * blockDim.x*2 + threadIdx.x;
				
	if(id < image_size){
		if((id+blockDim.x) >= image_size){
			val[tid] = normM_c[id];
		}
		else{
			if(normM_c[id]>normM_c[id + blockDim.x]){
				val[tid]=normM_c[id];
			}
			else{
				val[tid]=normM_c[id + blockDim.x];
			}
		}
	}
	else{
		val[tid] = -1;
	}
	
	__syncthreads();
	
	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1){
		if (tid < s){
			if(val[tid]<=val[tid+s]){
				val[tid]=val[tid+s];
			}
		}
		__syncthreads();
	}
	d_projections[blockIdx.x]=val[0];
	
	__syncthreads();
}



__global__ void maxValExtract(float *normM_c, float *normM1_c, long int image_size, float *d_projections, int *d_index, float a){ 
	__shared__ int pos[2048];
	__shared__ float val[2048];
	
	unsigned int tid = threadIdx.x;
    unsigned int id = blockIdx.x*2 * blockDim.x + threadIdx.x;
	float faux, faux2;
	
	faux = ((a - normM_c[id])/a);
	faux2 = ((a - normM_c[id + blockDim.x])/a);
		
	
	if(id < image_size && faux <= 1.0e-6){
		val[tid] = normM1_c[id];
		pos[tid] = id;
		
	}
	else{
		val[tid] = -1;
	}
	
	if(id + blockDim.x < image_size && faux2 <= 1.0e-6){
		val[tid + blockDim.x] = normM1_c[id + blockDim.x];
		pos[tid + blockDim.x] = id + blockDim.x;		
	}
	else{
		val[tid + blockDim.x] = -1;
	}
	__syncthreads();
	
	for (unsigned int s = blockDim.x; s > 0; s>>=1){
		if (tid < s){
			if(val[tid]<=val[tid+s]){
				val[tid] = val[tid+s];
				pos[tid] = pos[tid+s];
			}			
		}
		__syncthreads();
	}
	
	d_projections[blockIdx.x]=val[0];
	d_index[blockIdx.x]=(int)pos[0];


	__syncthreads();	
}


__global__ void actualizacion(float *v_c, float *image_c, int bands, float *normM_c, long int image_size) { 
	__shared__ float block_v[DIM];
	int k, i;
	float faux = 0;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
		
		
	if(blockDim.x < bands){
		for(i = threadIdx.x; i < bands; i += blockDim.x){
			block_v[i] = v_c[i];
		}
	}
	else{
		if(threadIdx.x < bands){
			block_v[threadIdx.x] = v_c[threadIdx.x];
		}
	}
	__syncthreads();
	
	if (j < image_size){	
		faux = 0;
		for(k = 0; k < bands; k++){
			faux += block_v[k] * image_c[k*image_size + j];
		}
		normM_c[j] -= faux * faux;
	}
}


__global__ void normalizacion(float *image_c, int bands, long int image_size, float *normM_c, float *normM1_c) { 
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
		normM1_c[i] = aux;
	}

}

__global__ void calculateNormM(float *image_c, int bands, long int image_size, float *normM_c, float *normM1_c) { 
	int k;
	int j =  blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < image_size){
		for(k = 0; k < bands; k++){
				normM_c[j] += image_c[k*image_size + j] * image_c[k*image_size + j]; 
				normM1_c[j] += image_c[k*image_size + j] * image_c[k*image_size + j];
			}
	}

}

void checkCUDAError(const char *mensaje, cudaError_t error){
	
	if(error != cudaSuccess){
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error), mensaje);
	
	}
}

void reservarMemoria(int bands,long int image_size, float **v_c, float **image_c, float **normM_c, float **normM1_c, float **image, float **v, float **d_projections, int **d_index, float **h_projections, int **h_index, int globalSize_reduction){	
	cudaError_t error;
	
	error = cudaMalloc(v_c, bands*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de v_c", error);
	
	error = cudaMalloc(image_c, bands*image_size*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de image_c", error);
	
	error = cudaMalloc(normM_c, image_size*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de normMc", error);
	
	error = cudaMalloc(normM1_c, image_size*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de normM1_c", error);
	
	error = cudaHostAlloc(image, bands*image_size*sizeof(float), cudaHostAllocDefault); 
	checkCUDAError("ERROR EN cudaHostAlloc de image", error);
	
	error = cudaHostAlloc(v, image_size*sizeof(float), cudaHostAllocDefault);
	checkCUDAError("ERROR EN cudaHostAlloc de v", error);
	
	error = cudaMalloc(d_projections, globalSize_reduction*sizeof(float));
	checkCUDAError("ERROR EN cudaMalloc de d_projections", error);
	
	error = cudaMalloc(d_index, globalSize_reduction*sizeof(int));
	checkCUDAError("ERROR EN cudaMalloc de d_projections", error);
	
	error = cudaHostAlloc(h_projections, globalSize_reduction*sizeof(float), cudaHostAllocDefault);
	checkCUDAError("ERROR EN cudaHostAlloc de d_projections", error);
	
	error = cudaHostAlloc(h_index, globalSize_reduction*sizeof(int), cudaHostAllocDefault);
	checkCUDAError("ERROR EN cudaHostAlloc de d_projections", error);
	

}


void liberarMemoria(float *v_c, float *image_c, float *normM_c, float *image, float *normM1_c, float *v, float *d_projections, int *d_index, float *h_projections, int *h_index){
	cudaError_t error;
	
	error = cudaFree(v_c);
	checkCUDAError("ERROR EN cudaFree de v_c", error);
	
	error = cudaFree(image_c);
	checkCUDAError("ERROR EN cudaFree de image_c", error);
	
	error = cudaFree(normM_c);
	checkCUDAError("ERROR EN cudaFree de normM_c", error);
	
	error = cudaFreeHost(image);
	checkCUDAError("ERROR EN cudaFreeHost de image", error);
	
	error = cudaFree(normM1_c);
	checkCUDAError("ERROR EN cudaFreeHost de normM1_c", error);
	
	error = cudaFreeHost(v);
	checkCUDAError("ERROR EN cudaFreeHost de v", error);
	
	error = cudaFree(d_projections);
	checkCUDAError("ERROR EN cudaFree de d_projections", error);

	error = cudaFree(d_index);
	checkCUDAError("ERROR EN cudaFree de d_index", error);	
	
	error = cudaFreeHost(h_projections);
	checkCUDAError("ERROR EN cudaFreeHost de h_projections", error);
	
	error = cudaFreeHost(h_index);
	checkCUDAError("ERROR EN cudaFreeHost de h_index", error);	
	
	
}

void selectDevice(){
	int count;
	int	i , device;
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


void actualizarNormM(float *v, int bands, long int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c){
	cudaError_t error;
	int val = ceil((double)image_size/1024);
		
	error = cudaMemcpy(v_c, v, bands*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("ERROR EN cudamemcpy de v_c", error);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(val); 
	
	
	actualizacion<<<dimGrid,dimBlock>>>(v_c, image_c, bands, normM_c, image_size);
	checkCUDAError("ERROR EN kernel actualización", cudaGetLastError());
	cudaDeviceSynchronize();	
	
	
	
}

void normalizeImgC(float *image, long int image_size, int bands,float *image_c, int rows, float *normM_c, float *normM1_c){
	cudaError_t error;
	int val = ceil((double)image_size/1024);

	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice); 
	checkCUDAError("ERROR EN cudaMemcpy de image_c", error);
	
		
	dim3 dimBlock(1024);
	dim3 dimGrid(val); 

	normalizacion<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c, normM1_c);
	checkCUDAError("ERROR EN kernel normalización", cudaGetLastError());
	cudaDeviceSynchronize();


}

void calculateNormM(float *image, long int image_size, int bands, int rows, float *image_c, float *normM_c, float *normM1_c){
	cudaError_t error;
	int val = ceil((double)image_size/1024);
	
	error = cudaMemcpy(image_c, image, bands*image_size*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("ERROR EN cudaMemcpy de image_c", error);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(val); 
	
	calculateNormM<<<dimGrid,dimBlock>>>(image_c, bands, image_size, normM_c, normM1_c);
	checkCUDAError("ERROR EN kernel calculateNormM", cudaGetLastError());
	cudaDeviceSynchronize();
	

}


void calculateMaxVal(int image_size, float *normM_c, float *d_projections, float *h_projections){
	
	int val = ceil((double)image_size/2/1024);
	cudaError_t error;
	
	dim3 dimBlock(1024);
	dim3 dimGrid(val); 

	
	maxVal<<<dimGrid,dimBlock>>>(normM_c, image_size, d_projections);
	checkCUDAError("ERROR EN kernel maxVal", cudaGetLastError());
	cudaDeviceSynchronize();	
	
	error = cudaMemcpy(h_projections, d_projections, val*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudaMemcpy de h_projections", error);


}


void calculateMaxValExtract_2(int image_size, float *normM_c, float *normM1_c, float *d_projections, float *h_projections, int *d_index, int *h_index, float a){
	
	cudaError_t error;
	int val = ceil((double)image_size/2/1024);
	
	dim3 dimBlock(1024);
	dim3 dimGrid(val); 
	
	maxValExtract<<<dimGrid,dimBlock>>>(normM_c, normM1_c, image_size, d_projections, d_index, a);
	checkCUDAError("ERROR EN kernel maxValExtract", cudaGetLastError());
	cudaDeviceSynchronize();	
	
	error = cudaMemcpy(h_projections, d_projections, val*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudaMemcpy de h_projections", error);
	
	error = cudaMemcpy(h_index, d_index, val*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError("ERROR EN cudaMemcpy de h_index", error);

}


