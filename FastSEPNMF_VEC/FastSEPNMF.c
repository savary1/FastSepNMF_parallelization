//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ReadWrite.h"

void normalizeImg(float *image, long int image_size, int bands);
float maxVal(float *vector, long int image_size);

int main (int argc, char* argv[]) {

	struct timeval t0, t_fin, t1, t2;
	float secs_end, t_sec, t_usec, t_norm, t_cost_loop, t_seq;
	int rows, cols, bands; //size of the image
	int datatype;
	int endmembers;
	int normalize;
	long int i, j, d, k;
	float max_val, max_min_val, a, b, faux, faux2;

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

	secs_end = t_sec = t_usec = t_norm = t_cost_loop = t_seq = 0;

	/**************************** #INIT# - Load Image and allocate memory******************************/
	//read image header
	readHeader(argv[2], &cols, &rows, &bands, &datatype);
	printf("\nLines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", rows, cols, bands, datatype);


	long int image_size = cols*rows;
	float *image = (float *) calloc (image_size * bands, sizeof(float));    	//input image
	float *U = (float *) malloc (bands * endmembers * sizeof(float));       	//selected endmembers
	float *normM = (float *) calloc (image_size, sizeof(float));            	//normalized image
	float *v = (float *) malloc (bands * sizeof(float));						//used to update normM in every iteration                                                    	//float auxiliary array
	long int J[endmembers];                                                 	//selected endmembers positions in input image

	Load_Image(argv[1], image, cols, rows, bands, datatype);
	/**************************** #END# - Load Image and allocate memory*******************************/

	gettimeofday(&t0,NULL);

	/**************************** #INIT# - Normalize image****************************************/
	gettimeofday(&t1,NULL);
	if (normalize == 1) {
		normalizeImg(image, image_size, bands);
	}

	for(i = 0; i < image_size; i++) {
		for(k = 0; k < bands; k++) {
			normM[i] += image[i*bands + k] * image[i*bands + k];
		}
	}

	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	t_norm = t_sec + t_usec/1.0e+6;

	/**************************** #END# - Normalize image****************************************/
	max_val = maxVal(normM, image_size);
	/**************************** #INIT# - FastSEPNMF algorithm****************************************/

	i = 0;
	//while i <= r && max(normM)/nM > 1e-9
	while(i < endmembers) {
		gettimeofday(&t1,NULL);
		//[a,b] = max(normM);
		a = maxVal(normM, image_size);

		if(a/max_val <= 1e-9) {
			break;
		}

		//(a-normM)/a
		//b = find((a-normM)/a <= 1e-6);
		max_min_val = -1;
		for(j = 0; j < image_size; j++) {
			faux = (a - normM[j])/a;
			if (faux <= 1.0e-6 && faux > max_min_val) {
				J[i] = j;
				max_min_val = faux;
			}
		}

		//U(:,i) = M(:,b);
		for(j = 0 ; j < bands; j++) {
			U[i*bands + j] = image[J[i]*bands + j];
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
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		t_seq = t_seq + t_sec + t_usec/1.0e+6;

		//(v'*M).^2
		//normM = normM - (v'*M).^2;
		gettimeofday(&t1,NULL);		
		for(j = 0; j < image_size; j++) {
			faux = 0;
			for(k = 0; k < bands; k++) {
				faux += v[k] * image[j*bands + k];
			}
			normM[j] -= faux * faux;
		}
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
	secs_end = t_sec + t_usec/1.0e+6;

	printf("Endmembers:\n");
	for(i = 0; i < endmembers; i++) {
		printf("%ld \t- %ld \t- Coordenadas: (%ld,%ld) \n", i, J[i],(J[i] / cols),(J[i] % cols));
	}

	printf("Total time:	\t%.5f segundos\n", secs_end);
	printf("T norm:	\t\t%.5f segundos\n", t_norm);
	printf("T square loop:	\t%.5f segundos\n", t_cost_loop);
	printf("T sequential:	\t%.5f segundos\n", t_seq);


	free(image);
	free(U);
	free(normM);
	free(v);

	return 0;
}


float maxVal(float *vector, long int image_size) {
	float max_val = -1;
	long int i;

	for (i = 0; i < image_size; i++) {
		if(vector[i] > max_val) {
			max_val = vector[i];
		}
	}

	return max_val;
}


void normalizeImg(float *image, long int image_size, int bands) {
	long int i, j, row;
	float norm_val;
	
	for (i = 0; i < image_size ; i++) {
		row = i*bands;
		norm_val = 0;
		for(j = 0; j < bands; j++) {
			norm_val += image[row + j];
		}
		norm_val = 1.0/(norm_val + 1.0e-16);
		for(j = 0; j < bands; j++) {
			image[row + j] = image[row + j] * norm_val;
		}
	}
}
