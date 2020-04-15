//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ReadWrite.h"

#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable



void normalize_img(float *image, long int image_size, int bands);
void normalize_img_IIR(float *image_device, long int image_size, int bands);
long int max_val_extract_array(float *normMAux, long int *b_pos, long int b_pos_size);
float max_Val(float *vector, long int image_size);
void exit_if_OpenCL_fail(cl_int code, char* msg);
cl_device_id select_device();
cl_program build_kernels(cl_context clContext, cl_device_id selectedDevice);



int main (int argc, char* argv[]){

	struct timeval t0, tfin, t1, t2;
	float secsFin, t_sec, t_usec, tNorm, tCostSquare;
    int rows, cols, bands; //size of the image
    int datatype, endmembers, normalize, deviceSelected;
    long int i, j, b_pos_size, d, k;
    float max_val, a, b, faux, faux2;

	cl_int status;
	cl_device_id selectedDevice;
	cl_context clContext;
	cl_command_queue clQueue;
	cl_program clProgram;
	cl_kernel updateNormMKernel, normalizeImgKernel;
	cl_mem clImage, clV, clNormM, clImageHost;
	

    if (argc != 5){
		printf("******************************************************************\n");
		printf("	ERROR in the input parameters:\n");
		printf("	The correct sintax is:\n");
		printf("	./atdca_serie image.bsq image.hdr numEndmembers normalize        \n");
		printf("******************************************************************\n");
		return(0);
	}
	else {
		// PARAMETERS.
		endmembers = atoi(argv[3]);
        normalize = atoi(argv[4]);
	}


	/************************************* #INIT# - OpenCL init****************************************/
	
    selectedDevice = select_device();

	clContext = clCreateContext(NULL, 1, &selectedDevice, NULL, NULL, &status);
	exit_if_OpenCL_fail(status, "clCreateContext returned error");

	clQueue = clCreateCommandQueue(clContext, selectedDevice, CL_QUEUE_PROFILING_ENABLE, &status);
	exit_if_OpenCL_fail(status, "clCreateContext returned error");

	clProgram = build_kernels(clContext, selectedDevice);

	updateNormMKernel = clCreateKernel(clProgram, "update_normM", &status);
	exit_if_OpenCL_fail(status, "Error creating update_normM kernel");

	/************************************* #END# - OpenCL init****************************************/


	secsFin = t_sec = t_usec = tNorm = tCostSquare = 0;

    /**************************** #INIT# - Load Image and allocate memory******************************/
    //reading image header
  	readHeader(argv[2], &cols, &rows, &bands, &datatype);
    printf("\nLines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", rows, cols, bands, datatype);
	
	int image_size = cols*rows;
    float *image = (float *) malloc (image_size * bands * sizeof(float));    	//input image
    float *U = (float *) malloc (bands * endmembers * sizeof(float));       	//selected endmembers
    float *normM = (float *) calloc (image_size, sizeof(float));            	//normalized image
	float *normM1 = (float *) malloc (image_size * sizeof(float));				//copy of normM
	float *normMAux = (float *) malloc (image_size * sizeof(float));			//aux array to find the positions of (a-normM)/a <= 1e-6
	long int *b_pos = (long int *) malloc (image_size * sizeof(long int));	   	//valid positions of normM that meet (a-normM)/a <= 1e-6
	float *v = (float *) malloc (bands * sizeof(float));						//used to update normM in every iteration
    long int J[endmembers];                                                 	//selected endmembers positions in input image

	Load_Image_IIR(argv[1], image, image_size, bands, datatype);

	size_t localSize = 1024;
	size_t globalsize = ceil(image_size/(float)localSize) * localSize;

	clImage = clCreateBuffer(clContext, CL_MEM_READ_WRITE, image_size * bands * sizeof(float), NULL, &status);
	exit_if_OpenCL_fail(status, "Error creating clImage buffer on device");
	clImageHost = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, image_size * bands * sizeof(float), NULL, &status);
	exit_if_OpenCL_fail(status, "Error creating clImageHost buffer on device");
	clNormM = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, image_size * sizeof(float), NULL, &status);
	exit_if_OpenCL_fail(status, "Error creating clNorm buffer on device");
	clV = clCreateBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bands * sizeof(float), v, &status);
	exit_if_OpenCL_fail(status, "Error creating clV buffer on device");

	

	/**************************** #END# - Load Image and allocate memory*******************************/

	gettimeofday(&t0,NULL);

    /**************************** #INIT# - Normalize image****************************************/
	status = clEnqueueWriteBuffer(clQueue, clImage, CL_TRUE, 0, image_size * bands * sizeof(float), image, 0, NULL, NULL);
	exit_if_OpenCL_fail(status, "Error copying the image to the device");

	gettimeofday(&t1,NULL);
    if (normalize == 1){
        normalizeImgKernel = clCreateKernel(clProgram, "normalize_img", &status);
		exit_if_OpenCL_fail(status, "Error creating normalize_img kernel");

		status  = clSetKernelArg(normalizeImgKernel, 0, sizeof(cl_mem), &clImage);
		exit_if_OpenCL_fail(status, "Error setting image as parameter in the device");
		status  = clSetKernelArg(normalizeImgKernel, 1, sizeof(cl_mem), &clImageHost);
		exit_if_OpenCL_fail(status, "Error setting imageHost as parameter in the device");
		status  = clSetKernelArg(normalizeImgKernel, 2, sizeof(cl_mem), &clNormM);
		exit_if_OpenCL_fail(status, "Error setting normM as parameter in the device");
		status  = clSetKernelArg(normalizeImgKernel, 3, sizeof(int), &image_size);
		exit_if_OpenCL_fail(status, "Error setting image_size as parameter in the device");
		status  = clSetKernelArg(normalizeImgKernel, 4, sizeof(int), &bands);
		exit_if_OpenCL_fail(status, "Error setting bands as parameter in the device");

		status = clEnqueueNDRangeKernel(clQueue, normalizeImgKernel, 1, NULL, &globalsize, &localSize, 0, NULL, NULL);
		exit_if_OpenCL_fail(status, "Error executing kernel");
		clFinish(clQueue);

		// status = clEnqueueReadBuffer(clQueue, clImageHost, CL_TRUE, 0, image_size * bands * sizeof(float), image, 0, NULL, NULL);
		// exit_if_OpenCL_fail(status, "Error reading normM from the device");
		status = clEnqueueReadBuffer(clQueue, clNormM, CL_TRUE, 0, image_size * sizeof(float), normM, 0, NULL, NULL);
		exit_if_OpenCL_fail(status, "Error reading normM from the device");
		clFinish(clQueue);
    }
	else{
		//Este for se puede separa en 2, el de fuera de longitud image size y el de dentro vectorizarlo
		#pragma omp parallel for
		for(i = 0; i < image_size; i++){
			for(k = 0; k < bands; k++){
				normM[i] += image[i*bands + k] * image[i*bands + k]; 
			}
		}
	}

	
	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
  	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	tNorm = t_sec + t_usec/1.0e+6;

	/**************************** #END# - Normalize image****************************************/
	max_val = max_Val(normM, image_size);
    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
	//if i == 1, normM1 = normM; 
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}
	

	status  = clSetKernelArg(updateNormMKernel, 0, sizeof(cl_mem), &clImage);
	exit_if_OpenCL_fail(status, "Error setting image as parameter in the device");
	status  = clSetKernelArg(updateNormMKernel, 1, sizeof(cl_mem), &clV);
	exit_if_OpenCL_fail(status, "Error setting v as parameter in the device");
	status  = clSetKernelArg(updateNormMKernel, 2, sizeof(cl_mem), &clNormM);
	exit_if_OpenCL_fail(status, "Error setting normM as parameter in the device");
	status = clSetKernelArg(updateNormMKernel, 3, sizeof(int), &bands);
	exit_if_OpenCL_fail(status, "Error setting bands as parameter in the device");
	status = clSetKernelArg(updateNormMKernel, 4, sizeof(int), &image_size);
	exit_if_OpenCL_fail(status, "Error setting image_size as parameter in the device");
	

	i = 0;
	//while i <= r && max(normM)/nM > 1e-9
	while(i < endmembers){
		//[a,b] = max(normM);
		a = max_Val(normM, image_size);

		if(a/max_val <= 1e-9){
			break;
		}

		//(a-normM)/a
		for(j = 0; j < image_size; j++){
			normMAux[j] = (a - normM[j])/a;
		}

		//b = find((a-normM)/a <= 1e-6);
		b_pos_size = 0;
		for(j = 0; j < image_size; j++){
			if (normMAux[j]<= 1.0e-6){
				b_pos[b_pos_size] = j;
				b_pos_size++;
			}
		}
		
		//if length(b) > 1, [c,d] = max(normM1(b)); b = b(d);
		if (b_pos_size > 1){
			d = max_val_extract_array(normM1, b_pos, b_pos_size);
			b = b_pos[d];
			J[i] = b;
		}
		else{ // comprobar si siempre tiene valores b_pos
			J[i] = b_pos[0];
		}
		
		//U(:,i) = M(:,b);  //MIRAR SI SE PUEDEN HACER LOS ACCESOS A MEMORIA ADYACENTES
		for(j = 0 ; j < bands; j++){
			//U[i*bands + j] = image[J[i] * bands + j];
			U[i*bands + j] = image[image_size*j + J[i]];
		}
		
		//U(:,i) = U(:,i) - U(:,j)*(U(:,j)'*U(:,i));
		for(j = 0; j < i; j++){
			faux = 0;
			//(U(:,j)'*U(:,i))
			for(k = 0; k < bands; k++){//MIRAR SI LOS ACCESOS DE PUEDEN HACER ADYACENTES
				faux += U[j*bands + k] * U[i*bands + k];
			}
			
			//MIRAR SI LOS ACCESOS A MEMORIA SE PUEDEN HACER ADYACENTES
			#pragma ivdep
			for(k = 0; k < bands; k ++){
				faux2 = U[j*bands + k] * faux;
				U[i*bands + k] = U[i*bands + k] - faux2;
			}		
		}
		
		//U(:,i) = U(:,i)/norm(U(:,i));
		//v = U(:,i);
		faux = 0;
		for(j = 0; j < bands; j++){//INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
			faux += U[i*bands + j]*U[i*bands + j];
		}
		faux = sqrt(faux);
		for(j = 0; j < bands; j++){	//NO LO VECTORIZA, CREO QUE SI SE PUEDE. INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
			U[i*bands + j] = U[i*bands + j]/faux;
			v[j] = U[i*bands + j];
		}

		// for j = i-1 : -1 : 1
		for(j = i - 1; j >= 0; j--){
			//(v'*U(:,j))
			faux = 0;
			for(k = 0; k < bands; k++){//INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
				faux += v[k] * U[j*bands + k];
			}
			//(v'*U(:,j))*U(:,j);//HACER ACCESOS A MEMORIA ADYACENTES
			//v = v - (v'*U(:,j))*U(:,j);
			for(k = 0; k < bands; k ++){
				faux2 = U[j*bands + k] * faux;
				v[k] = v[k] - faux2;
			}
		}

		gettimeofday(&t1,NULL);
		
		status = clEnqueueWriteBuffer(clQueue, clV, CL_TRUE, 0, bands * sizeof(float), v, 0, NULL, NULL);
		exit_if_OpenCL_fail(status, "Error copying v to the device");

		printf("Ejecutando kernel - %d\n", i);
		status = clEnqueueNDRangeKernel(clQueue, updateNormMKernel, 1, NULL, &globalsize, &localSize, 0, NULL, NULL);
		exit_if_OpenCL_fail(status, "Error executing kernel");
		clFinish(clQueue);

		status = clEnqueueReadBuffer(clQueue, clNormM, CL_TRUE, 0, image_size * sizeof(float), normM, 0, NULL, NULL);
		exit_if_OpenCL_fail(status, "Error reading normM from the device");
		clFinish(clQueue);

		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tCostSquare = tCostSquare + t_sec + t_usec/1.0e+6;

		
			
		i = i + 1;
		
	}
	/**************************** #END# - FastSEPNMF algorithm*****************************************/
	
	gettimeofday(&tfin,NULL);
	t_sec  = (float)  (tfin.tv_sec - t0.tv_sec);
  	t_usec = (float)  (tfin.tv_usec - t0.tv_usec);
	secsFin = t_sec + t_usec/1.0e+6;

	printf("Endmembers:\n");
    for(i = 0; i < endmembers; i++){
		printf("%ld \t- %ld \t- Coordenadas: (%ld,%ld) \t- Valor: %f\n", i, J[i],(J[i] / cols),(J[i] % cols), normM1[J[i]]);
    }

	printf("Total time:	\t%.5f segundos\n", secsFin);
	printf("T norm:	\t\t%.5f segundos\n", tNorm);
	printf("T square loop:	\t%.5f segundos\n", tCostSquare);

	clReleaseMemObject(clImage);
   	clReleaseMemObject(clNormM);
	clReleaseMemObject(clV);
   	clReleaseKernel(updateNormMKernel);
   	clReleaseProgram(clProgram);
	clReleaseContext(clContext);
   	clReleaseCommandQueue(clQueue);

    free(image);
    free(U);
    free(normM);
	free(normM1);
	free(normMAux);
	free(b_pos);
	free(v);

    return 0;
}



long int max_val_extract_array(float *normMAux, long int *b_pos, long int b_pos_size){
	float max_val = -1;
	long int pos = -1;
	long int i;
    for (i = 0; i < b_pos_size; i++){ //DICE QUE NO SE PUEDE VECTORIZAR PORQUE MAXVAL DEPENDE DEL MAXVAL DE LA ITERACIÓN ANTERIOR
        if(normMAux[b_pos[i]] > max_val){
            max_val = normMAux[b_pos[i]];
			pos = i;
        }
    }
	return pos;
}



float max_Val(float *vector, long int image_size){
	float max_val = -1;
	long int i;

    for (i = 0; i < image_size; i++){
        if(vector[i] > max_val){
            max_val = vector[i];
        }
    }

	return max_val;
}



void exit_if_OpenCL_fail(cl_int code, char* msg){
	if(code != CL_SUCCESS){
		printf("Error: %s\n", msg);
		printf("Error code: %d\n", code);
		exit(1);
	}
}



cl_device_id select_device(){
	cl_int status;
	cl_uint numPlatforms, numDevices, deviceNumInfo;
	cl_ulong deviceLongInfo;
	size_t infoSize, localWorkSize;
	char *platformName, *deviceInfo;
	int i, selectedPlatform, selectedDevice;

	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	exit_if_OpenCL_fail(status, "clGetPlatformIDs returned error");
	printf("\nAvailable OpenCL platforms:\n");

	cl_platform_id platformIDs[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
	exit_if_OpenCL_fail(status, "clGetPlatformIDs returned error");

	for(i = 0; i < numPlatforms; i++){
		status = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
		exit_if_OpenCL_fail(status, "clGetPlatformInfo returned error");
		
		platformName = (char*)alloca(sizeof(char)*infoSize);
		status = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, infoSize, platformName, NULL);
		exit_if_OpenCL_fail(status, "clGetPlatformInfo returned error");
		printf("\t - Platform %d: %s\n", i, platformName);		
	}

	printf("\nSelect a platform: ");
	scanf("%d", &selectedPlatform);

	if(selectedPlatform > numPlatforms - 1)
		exit_if_OpenCL_fail(CL_DEVICE_NOT_AVAILABLE, "Platform number is not valid");
	
	status = clGetDeviceIDs(platformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	exit_if_OpenCL_fail(status, "clGetDeviceIDs returned error");

	cl_device_id deviceIDs[numDevices];
	status = clGetDeviceIDs(platformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, numDevices, deviceIDs, NULL);
	exit_if_OpenCL_fail(status, "clGetDeviceIDs returned error");

	printf("\nChoose a device from the selected platform:\n");
	for(i = 0; i < numDevices; i++){
		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 0, NULL, &infoSize);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		deviceInfo = (char*)alloca(sizeof(char)*infoSize);

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, infoSize, deviceInfo, NULL);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		printf("\t - Device %d: %s", i, deviceInfo);	

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceNumInfo), &deviceNumInfo, NULL);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		printf("  CU: %u", deviceNumInfo);

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceLongInfo), &deviceLongInfo, NULL);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		printf("  Local Memory: %u KB", (unsigned int) (deviceLongInfo/1024));


		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceLongInfo), &deviceLongInfo, NULL);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		printf("  Global Memory: %u MB\n", (unsigned int) (deviceLongInfo/1e6));

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &localWorkSize, NULL);
		exit_if_OpenCL_fail(status, "clGetDeviceInfo returned error");
		printf("  Max work group size: %zd\n", localWorkSize);
	}

	printf("\nChoose a device: ");
	scanf("%d", &selectedDevice);

	if(selectedDevice > numDevices - 1)
		exit_if_OpenCL_fail(CL_DEVICE_NOT_AVAILABLE, "Device number is not valid");

	return deviceIDs[selectedDevice];
}



cl_program build_kernels(cl_context clContext, cl_device_id selectedDevice) {
	cl_program clProgram;
	cl_int status, buildLogStatus;
	FILE *f;
	char *sourceCode, *logBuff;
	long size;
	size_t logSize;

	f = fopen("kernels.cl", "r");
	if(f == NULL)
		exit_if_OpenCL_fail(CL_BUILD_PROGRAM_FAILURE, "Error opening the file containing the kernels");

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	rewind(f);

	sourceCode = malloc(sizeof(char) * (size + 1));
	fread(sourceCode, 1, size, f);
	sourceCode[size] = '\0';

	clProgram = clCreateProgramWithSource(clContext, 1, (const char **) &sourceCode, NULL, &status);
	exit_if_OpenCL_fail(status, "Error creating cumputing program");

	status = clBuildProgram(clProgram, 1, &selectedDevice, NULL, NULL, NULL);
		buildLogStatus = clGetProgramBuildInfo(clProgram, selectedDevice, CL_PROGRAM_BUILD_LOG, NULL, NULL, &logSize);
		exit_if_OpenCL_fail(buildLogStatus, "Error getting kernel build logs (1)");
		logBuff = (char *) malloc(logSize);
		buildLogStatus = clGetProgramBuildInfo(clProgram, selectedDevice, CL_PROGRAM_BUILD_LOG, logSize, logBuff, NULL);
		exit_if_OpenCL_fail(buildLogStatus, "Error getting kernel build logs (2)");
		printf("Kernel build log:\n %s", logBuff);
	exit_if_OpenCL_fail(status, "Error building computing program");
	
	close(f);
	free(sourceCode);
	free(logBuff);

	return clProgram;
}