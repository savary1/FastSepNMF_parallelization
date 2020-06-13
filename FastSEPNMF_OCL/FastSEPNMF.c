//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ReadWrite.h"



void exitIfOpenCLFail(cl_int code, char* msg);
cl_device_id selectDevice();
cl_program buildKernels(cl_context cl_context, cl_device_id selected_device);



int main (int argc, char* argv[]) {

	struct timeval t0, t_fin, t1, t2, t3, t4;
	float secs_fin, t_sec, t_usec, t_norm, t_cost_loop;
	int rows, cols, bands; //size of the image
	int datatype;
	int endmembers;
	int normalize;
	long int i, j, b_pos_size, d, k;
	float max_val, a, b, faux, faux2, max_red;

	cl_int status;
	cl_device_id selected_device;
	cl_context cl_context;
	cl_command_queue cl_queue;
	cl_program cl_program;
	cl_kernel update_normM_kernel, normalize_img_kernel, initialize_normM, normM_reduction_kernel, select_endmember_kernel;
	cl_mem cl_image, cl_v, cl_normM, cl_normM1, cl_red_result, cl_red_result_pos;

	if (argc != 5) {
		printf("******************************************************************\n");
		printf("	ERROR in the input parameters:\n");
		printf("	The correct sintax is:\n");
		printf("	./FastSEPNMF image.bsq image.hdr numEndmembers normalize          \n");
		printf("******************************************************************\n");
		return(0);
	} else {
		// parameters
		endmembers = atoi(argv[3]);
		normalize = atoi(argv[4]);
	}

	secs_fin = t_sec = t_usec = t_norm = t_cost_loop = 0;

	/************************************* #INIT# - OpenCL init****************************************/

	selected_device = selectDevice();

	cl_context = clCreateContext(NULL, 1, &selected_device, NULL, NULL, &status);
	exitIfOpenCLFail(status, "clCreateContext returned error");

	cl_queue = clCreateCommandQueue(cl_context, selected_device, CL_QUEUE_PROFILING_ENABLE, &status);
	exitIfOpenCLFail(status, "clCreateCommandQueue returned error");

	cl_program = buildKernels(cl_context, selected_device);

	update_normM_kernel = clCreateKernel(cl_program, "update_normM", &status);
	exitIfOpenCLFail(status, "Error creating update_normM kernel");

	normM_reduction_kernel = clCreateKernel(cl_program, "normM_reduction", &status);
	exitIfOpenCLFail(status, "Error creating normM_reduction kernel");

	select_endmember_kernel = clCreateKernel(cl_program, "select_endmember", &status);
	exitIfOpenCLFail(status, "Error creating select_endmember kernel");

	/************************************* #END# - OpenCL init****************************************/


	/**************************** #INIT# - Load Image and allocate memory******************************/
	//reading image header
	readHeader(argv[2], &cols, &rows, &bands, &datatype);
	printf("\nLines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", rows, cols, bands, datatype);


	long int image_size = cols*rows;
	float *image = (float *) calloc (image_size * bands, sizeof(float));    	//input image
	float *U = (float *) malloc (bands * endmembers * sizeof(float));       	//selected endmembers
	float *normM;																//normalized image
	float *v;                                                  					//float auxiliary array
	float *red_result;
	int *red_result_pos;
	long int J[endmembers];                                                 	//selected endmembers positions in input image


	Load_Image_IIR(argv[1], image, image_size, bands, datatype);

	gettimeofday(&t0,NULL);

	size_t localSize = 1024;
	size_t globalsize = ceil(image_size/(float)localSize) * localSize;
	size_t reduction_size = ceil(image_size/2/(float)localSize);
	size_t global_reduction_size = ceil(image_size/2/(float)localSize) * localSize;

	cl_image = clCreateBuffer(cl_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, image_size * bands * sizeof(float), image, &status);
	exitIfOpenCLFail(status, "Error creating cl_image buffer on device");
	cl_normM = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, image_size * sizeof(float), NULL, &status);
	exitIfOpenCLFail(status, "Error creating cl_normM buffer on device");
	cl_normM1 = clCreateBuffer(cl_context, CL_MEM_READ_WRITE, image_size * sizeof(float), NULL, &status);
	exitIfOpenCLFail(status, "Error creating cl_normM1 buffer on device");
	cl_v = clCreateBuffer(cl_context, CL_MEM_READ_ONLY, bands * sizeof(float), NULL, &status);
	exitIfOpenCLFail(status, "Error creating cl_v buffer on device");
	cl_red_result = clCreateBuffer(cl_context, CL_MEM_WRITE_ONLY, reduction_size * sizeof(float), NULL, &status);
	exitIfOpenCLFail(status, "Error creating cl_red_result buffer on device");
	cl_red_result_pos = clCreateBuffer(cl_context, CL_MEM_WRITE_ONLY, reduction_size * sizeof(int), NULL, &status);
	exitIfOpenCLFail(status, "Error creating cl_red_result_pos buffer on device");

	status  = clSetKernelArg(update_normM_kernel, 0, sizeof(cl_mem), &cl_image);
	exitIfOpenCLFail(status, "Error setting image as parameter in the device for update_normM_kernel");
	status  = clSetKernelArg(update_normM_kernel, 1, sizeof(cl_mem), &cl_v);
	exitIfOpenCLFail(status, "Error setting v as parameter in the device for update_normM_kernel");
	status  = clSetKernelArg(update_normM_kernel, 2, sizeof(cl_mem), &cl_normM);
	exitIfOpenCLFail(status, "Error setting normM as parameter in the device for update_normM_kernel");
	status = clSetKernelArg(update_normM_kernel, 3, sizeof(int), &bands);
	exitIfOpenCLFail(status, "Error setting bands as parameter in the device for update_normM_kernel");
	status = clSetKernelArg(update_normM_kernel, 4, sizeof(int), &image_size);
	exitIfOpenCLFail(status, "Error setting image_size as parameter in the device");

	status = clSetKernelArg(normM_reduction_kernel, 0, sizeof(cl_mem), &cl_normM);
	exitIfOpenCLFail(status, "Error setting normM as parameter in the device for normM_reduction_kernel");
	status = clSetKernelArg(normM_reduction_kernel, 1, sizeof(int), &image_size);
	exitIfOpenCLFail(status, "Error setting image_size as parameter in the device");
	status = clSetKernelArg(normM_reduction_kernel, 2, sizeof(cl_mem), &cl_red_result);
	exitIfOpenCLFail(status, "Error setting cl_red_result as parameter in the device for normM_reduction_kernel");

	status = clSetKernelArg(select_endmember_kernel, 0, sizeof(cl_mem), &cl_normM);
	exitIfOpenCLFail(status, "Error setting normM as parameter in the device for select_endmember_kernel");
	status = clSetKernelArg(select_endmember_kernel, 1, sizeof(cl_mem), &cl_normM1);
	exitIfOpenCLFail(status, "Error setting normM1 as parameter in the device for select_endmember_kernel");
	status = clSetKernelArg(select_endmember_kernel, 2, sizeof(int), &image_size);
	exitIfOpenCLFail(status, "Error setting image_size as parameter in the device");
	status = clSetKernelArg(select_endmember_kernel, 4, sizeof(cl_mem), &cl_red_result);
	exitIfOpenCLFail(status, "Error setting cl_red_result as parameter in the device for select_endmember_kernel");
	status = clSetKernelArg(select_endmember_kernel, 5, sizeof(cl_mem), &cl_red_result_pos);
	exitIfOpenCLFail(status, "Error setting cl_red_result_pos as parameter in the device for select_endmember_kernel");


	/**************************** #END# - Load Image and allocate memory*******************************/

	/**************************** #INIT# - Normalize image****************************************/

	gettimeofday(&t1,NULL);
	if (normalize == 1) {
		normalize_img_kernel = clCreateKernel(cl_program, "normalize_img", &status);
		exitIfOpenCLFail(status, "Error creating normalize_img_kernel kernel");

		status  = clSetKernelArg(normalize_img_kernel, 0, sizeof(cl_mem), &cl_image);
		exitIfOpenCLFail(status, "Error setting image as parameter in the device for normalize_img_kernel");
		status  = clSetKernelArg(normalize_img_kernel, 1, sizeof(cl_mem), &cl_normM);
		exitIfOpenCLFail(status, "Error setting normM as parameter in the device for normalize_img_kernel");
		status  = clSetKernelArg(normalize_img_kernel, 2, sizeof(cl_mem), &cl_normM1);
		exitIfOpenCLFail(status, "Error setting normM as parameter in the device for normalize_img_kernel");
		status  = clSetKernelArg(normalize_img_kernel, 3, sizeof(int), &image_size);
		exitIfOpenCLFail(status, "Error setting image_size as parameter in the device for normalize_img_kernel");
		status  = clSetKernelArg(normalize_img_kernel, 4, sizeof(int), &bands);
		exitIfOpenCLFail(status, "Error setting bands as parameter in the device for normalize_img_kernel");

		status = clEnqueueNDRangeKernel(cl_queue, normalize_img_kernel, 1, NULL, &globalsize, &localSize, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error executing normalize_img_kernel");
		clFinish(cl_queue);

		normM = (float *) clEnqueueMapBuffer(cl_queue, cl_normM, CL_TRUE, CL_MAP_READ, 0, image_size * sizeof(float), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping normM to the host");
		clFinish(cl_queue);

		clReleaseKernel(normalize_img_kernel);

	} else {
		initialize_normM = clCreateKernel(cl_program, "initialize_normM", &status);
		exitIfOpenCLFail(status, "Error creating initialize_normM kernel");

		status  = clSetKernelArg(initialize_normM, 0, sizeof(cl_mem), &cl_image);
		exitIfOpenCLFail(status, "Error setting image as parameter in the device for initialize_normM");
		status  = clSetKernelArg(initialize_normM, 1, sizeof(cl_mem), &cl_normM);
		exitIfOpenCLFail(status, "Error setting normM as parameter in the device for initialize_normM");
		status  = clSetKernelArg(initialize_normM, 2, sizeof(cl_mem), &cl_normM1);
		exitIfOpenCLFail(status, "Error setting normM as parameter in the device for initialize_normM");
		status  = clSetKernelArg(initialize_normM, 3, sizeof(int), &image_size);
		exitIfOpenCLFail(status, "Error setting image_size as parameter in the device for initialize_normM");
		status  = clSetKernelArg(initialize_normM, 4, sizeof(int), &bands);
		exitIfOpenCLFail(status, "Error setting bands as parameter in the device for initialize_normM");

		status = clEnqueueNDRangeKernel(cl_queue, initialize_normM, 1, NULL, &globalsize, &localSize, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error executing initialize_normM");
		clFinish(cl_queue);

		normM = (float *) clEnqueueMapBuffer(cl_queue, cl_normM, CL_TRUE, CL_MAP_READ, 0, image_size * sizeof(float), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping normM to the host");
		clFinish(cl_queue);

		clReleaseKernel(initialize_normM);
	}

	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	t_norm = t_sec + t_usec/1.0e+6;

	/**************************** #END# - Normalize image****************************************/
	status = clEnqueueNDRangeKernel(cl_queue, normM_reduction_kernel, 1, NULL, &global_reduction_size, &localSize, 0, NULL, NULL);
	exitIfOpenCLFail(status, "Error executing normM_reduction_kernel");
	clFinish(cl_queue);

	red_result = (float *) clEnqueueMapBuffer(cl_queue, cl_red_result, CL_TRUE, CL_MAP_READ, 0, reduction_size * sizeof(float), 0, NULL, NULL, &status);
	exitIfOpenCLFail(status, "Error mapping cl_red_result to the host");

	max_red = -1;
	for (j = 0; j < reduction_size; j++){
		if(red_result[j] > max_red)
			max_red = red_result[j];
	}

	status = clEnqueueUnmapMemObject(cl_queue, cl_red_result, red_result, 0, NULL, NULL);
	exitIfOpenCLFail(status, "Error unmapping red_result from the host");

	max_val = max_red;
	/**************************** #INIT# - FastSEPNMF algorithm****************************************/

	i = 0;
	//while i <= r && max(normM)/nM > 1e-9
	while(i < endmembers) {
		v = (float *) clEnqueueMapBuffer(cl_queue, cl_v, CL_TRUE, CL_MAP_WRITE, 0, bands * sizeof(float), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping v to the host");

		//[a,b] = max(normM);
		//a = maxVal(normM, image_size);
		status = clEnqueueNDRangeKernel(cl_queue, normM_reduction_kernel, 1, NULL, &global_reduction_size, &localSize, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error executing normM_reduction_kernel");
		clFinish(cl_queue);

		red_result = (float *) clEnqueueMapBuffer(cl_queue, cl_red_result, CL_TRUE, CL_MAP_READ, 0, reduction_size * sizeof(float), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping cl_red_result to the host");

		max_red = -1;
		for (j = 0; j < reduction_size; j++){
			if(red_result[j] > max_red)
				max_red = red_result[j];
		}

		status = clEnqueueUnmapMemObject(cl_queue, cl_red_result, red_result, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error unmapping red_result from the host");

		a = max_red;
		

		if(a/max_val <= 1e-9) {
			break;
		}

		//(a-normM)/a
		//b = find((a-normM)/a <= 1e-6);
		//if length(b) > 1, [c,d] = max(normM1(b)); b = b(d);
		status = clSetKernelArg(select_endmember_kernel, 3, sizeof(float), &max_red);
		exitIfOpenCLFail(status, "Error setting max_red as parameter in the device");

		status = clEnqueueNDRangeKernel(cl_queue, select_endmember_kernel, 1, NULL, &global_reduction_size, &localSize, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error executing select_endmember_kernel");
		clFinish(cl_queue);

		red_result = (float *) clEnqueueMapBuffer(cl_queue, cl_red_result, CL_TRUE, CL_MAP_READ, 0, reduction_size * sizeof(float), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping cl_red_result to the host");
		red_result_pos = (int *) clEnqueueMapBuffer(cl_queue, cl_red_result_pos, CL_TRUE, CL_MAP_READ, 0, reduction_size * sizeof(int), 0, NULL, NULL, &status);
		exitIfOpenCLFail(status, "Error mapping cl_red_result_pos to the host");

		max_red = -1;
		for (j = 0; j < reduction_size; j++){
			if(red_result[j] > max_red){
				J[i] = red_result_pos[j];
				max_red = red_result[j];
			}
		}

		status = clEnqueueUnmapMemObject(cl_queue, cl_red_result, red_result, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error unmapping red_result from the host");
		status = clEnqueueUnmapMemObject(cl_queue, cl_red_result_pos, red_result_pos, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error unmapping red_result_pos from the host");

		//U(:,i) = M(:,b);
		for(j = 0 ; j < bands; j++) {
			U[i*bands + j] = image[J[i] + image_size * j];
		}

		//U(:,i) = U(:,i) - U(:,j)*(U(:,j)'*U(:,i));
		for(j = 0; j < i; j++) {
			faux = 0;
			//(U(:,j)'*U(:,i))
			for(k = 0; k < bands; k++) {
				faux += U[j*bands + k] * U[i*bands + k];
			}

			#pragma ivdep
			for(k = 0; k < bands; k ++) {
				faux2 = U[j*bands + k] * faux;
				U[i*bands + k] = U[i*bands + k] - faux2;
			}

		}

		//U(:,i) = U(:,i)/norm(U(:,i));
		//v = U(:,i);
		faux = 0;
		for(j = 0; j < bands; j++) {
			faux += U[i*bands + j]*U[i*bands + j];
		}
		faux = sqrt(faux);
		for(j = 0; j < bands; j++) {
			U[i*bands + j] = U[i*bands + j]/faux;
			v[j] = U[i*bands + j];
		}

		// for j = i-1 : -1 : 1
		for(j = i - 1; j >= 0; j--) {
			//(v'*U(:,j))
			faux = 0;
			for(k = 0; k < bands; k++) {
				faux += v[k] * U[j*bands + k];
			}
			//(v'*U(:,j))*U(:,j);
			//v = v - (v'*U(:,j))*U(:,j);
			for(k = 0; k < bands; k ++) {
				faux2 = U[j*bands + k] * faux;
				v[k] = v[k] - faux2;
			}
		}

		//(v'*M).^2
		//normM = normM - (v'*M).^2;
		gettimeofday(&t1,NULL);
		status = clEnqueueUnmapMemObject(cl_queue, cl_v, v, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error unmapping v from the host");

		printf("Ejecutando kernel - %d\n", i);
		status = clEnqueueNDRangeKernel(cl_queue, update_normM_kernel, 1, NULL, &globalsize, &localSize, 0, NULL, NULL);
		exitIfOpenCLFail(status, "Error executing update_normM_kernel");
		clFinish(cl_queue);

		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		t_cost_loop = t_cost_loop + t_sec + t_usec/1.0e+6;

		i = i + 1;

	}
	/**************************** #END# - FastSEPNMF algorithm*****************************************/

	gettimeofday(&t_fin,NULL);
	t_sec  = (float)  (t_fin.tv_sec - t0.tv_sec);
	t_usec = (float)  (t_fin.tv_usec - t0.tv_usec);
	secs_fin = t_sec + t_usec/1.0e+6;

	printf("Endmembers:\n");
	for(j = 0; j < i; j++) {
		printf("%ld \t- %ld \t- Coordenadas: (%ld,%ld) \n", j, J[j],(J[j] / cols),(J[j] % cols));
	}

	printf("Total time:	\t%.5f segundos\n", secs_fin);
	printf("T norm:	\t\t%.5f segundos\n", t_norm);
	printf("T square loop:	\t%.5f segundos\n", t_cost_loop);

	clReleaseMemObject(cl_image);
	clReleaseMemObject(cl_normM);
	clReleaseMemObject(cl_normM1);
	clReleaseMemObject(cl_v);
	clReleaseMemObject(cl_red_result);
	clReleaseMemObject(cl_red_result_pos);
	clReleaseKernel(update_normM_kernel);
	clReleaseKernel(normM_reduction_kernel);
	clReleaseKernel(select_endmember_kernel);
	clReleaseProgram(cl_program);
	clReleaseCommandQueue(cl_queue);
	clReleaseContext(cl_context);


	free(image);
	free(U);

	return 0;
}



void exitIfOpenCLFail(cl_int code, char* msg) {
	if(code != CL_SUCCESS) {
		printf("Error: %s\n", msg);
		printf("Error code: %d\n", code);
		exit(1);
	}
}



cl_device_id selectDevice() {
	cl_int status;
	cl_uint num_platforms, num_devices, device_num_info;
	cl_ulong device_long_info;
	size_t info_size, local_work_size;
	char *platform_name, *device_info;
	int i, selected_platform, selected_device;

	status = clGetPlatformIDs(0, NULL, &num_platforms);
	exitIfOpenCLFail(status, "clGetPlatformIDs returned error");
	printf("\nAvailable OpenCL platforms:\n");

	cl_platform_id platformIDs[num_platforms];
	status = clGetPlatformIDs(num_platforms, platformIDs, NULL);
	exitIfOpenCLFail(status, "clGetPlatformIDs returned error");

	for(i = 0; i < num_platforms; i++) {
		status = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, 0, NULL, &info_size);
		exitIfOpenCLFail(status, "clGetPlatformInfo returned error");

		platform_name = (char*)alloca(sizeof(char)*info_size);
		status = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME, info_size, platform_name, NULL);
		exitIfOpenCLFail(status, "clGetPlatformInfo returned error");
		printf("\t - Platform %d: %s\n", i, platform_name);
	}

	printf("\nSelect a platform: ");
	scanf("%d", &selected_platform);

	if(selected_platform > num_platforms - 1)
		exitIfOpenCLFail(CL_DEVICE_NOT_AVAILABLE, "Platform number is not valid");

	status = clGetDeviceIDs(platformIDs[selected_platform], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	exitIfOpenCLFail(status, "clGetDeviceIDs returned error");

	cl_device_id deviceIDs[num_devices];
	status = clGetDeviceIDs(platformIDs[selected_platform], CL_DEVICE_TYPE_ALL, num_devices, deviceIDs, NULL);
	exitIfOpenCLFail(status, "clGetDeviceIDs returned error");

	printf("\nChoose a device from the selected platform:\n");
	for(i = 0; i < num_devices; i++) {
		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, 0, NULL, &info_size);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		device_info = (char*)alloca(sizeof(char)*info_size);

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_NAME, info_size, device_info, NULL);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		printf("\t - Device %d: %s", i, device_info);

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_num_info), &device_num_info, NULL);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		printf("  CU: %u", device_num_info);

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(device_long_info), &device_long_info, NULL);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		printf("  Local Memory: %u KB", (unsigned int) (device_long_info/1024));


		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_long_info), &device_long_info, NULL);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		printf("  Global Memory: %u MB\n", (unsigned int) (device_long_info/1e6));

		status = clGetDeviceInfo(deviceIDs[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &local_work_size, NULL);
		exitIfOpenCLFail(status, "clGetDeviceInfo returned error");
		printf("  Max work group size: %zd\n", local_work_size);
	}

	printf("\nChoose a device: ");
	scanf("%d", &selected_device);

	if(selected_device > num_devices - 1)
		exitIfOpenCLFail(CL_DEVICE_NOT_AVAILABLE, "Device number is not valid");

	return deviceIDs[selected_device];
}



cl_program buildKernels(cl_context cl_context, cl_device_id selected_device) {
	cl_program cl_program;
	cl_int status, build_log_status;
	FILE *f;
	char *source_code, *log_buff;
	long size;
	size_t log_size;

	f = fopen("kernels.cl", "r");
	if(f == NULL)
		exitIfOpenCLFail(CL_BUILD_PROGRAM_FAILURE, "Error opening the file containing the kernels");

	fseek(f, 0, SEEK_END);
	size = ftell(f);
	rewind(f);

	source_code = malloc(sizeof(char) * (size + 1));
	fread(source_code, 1, size, f);
	source_code[size] = '\0';

	cl_program = clCreateProgramWithSource(cl_context, 1, (const char **) &source_code, NULL, &status);
	exitIfOpenCLFail(status, "Error creating cumputing program");

	status = clBuildProgram(cl_program, 1, &selected_device, NULL, NULL, NULL);
	build_log_status = clGetProgramBuildInfo(cl_program, selected_device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &log_size);
	exitIfOpenCLFail(build_log_status, "Error getting kernel build logs (1)");
	log_buff = (char *) malloc(log_size);
	build_log_status = clGetProgramBuildInfo(cl_program, selected_device, CL_PROGRAM_BUILD_LOG, log_size, log_buff, NULL);
	exitIfOpenCLFail(build_log_status, "Error getting kernel build logs (2)");
	printf("Kernel build log:\n %s", log_buff);
	exitIfOpenCLFail(status, "Error building computing program");

	close(f);
	free(source_code);
	free(log_buff);

	return cl_program;
}