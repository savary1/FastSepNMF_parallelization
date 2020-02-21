//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ReadWrite.h"

void normalize_img(double *image, long int image_size, int bands);
long int max_val_extract_array(double *normMAux, long int *b_pos, long int b_pos_size);
double max_Val(double *vector, long int image_size);

int main (int argc, char* argv[]){

	struct timeval t0, tfin;
	float secsFin, t_sec, t_usec;
    int rows, cols, bands; //size of the image
    int datatype;
    int endmembers;
    int normalize;
    long int i, j, b_pos_size, d, k;
    double max_val, a, b, faux;

    if (argc != 5){
		printf("******************************************************************\n");
		printf("	ERROR in the input parameters:\n");
		printf("	The correct sintax is:\n");
		printf("	./atdca_serie image.bsq image.hdr numEndmembers normalize          \n");
		printf("******************************************************************\n");
		return(0);
	}
	else {
		// PARAMETERS.
		endmembers = atoi(argv[3]);
        normalize = atoi(argv[4]);
	}

    /**************************** #INIT# - Load Image and allocate memory******************************/
    //reading image header
  	readHeader(argv[2], &cols, &rows, &bands, &datatype);
    printf("\nLines = %d  Samples = %d  Nbands = %d  Data_type = %d\n", rows, cols, bands, datatype);


    long int image_size = cols*rows;
    double *image = (double *) calloc (image_size * bands, sizeof(double));    	//input image
    double *U = (double *) malloc (bands * endmembers * sizeof(double));       	//selected endmembers
    double *normM = (double *) calloc (image_size, sizeof(double));            	//normalized image
	double *normM1 = (double *) malloc (image_size * sizeof(double));			//copy of normM
	double *normMAux = (double *) malloc (image_size * sizeof(double));			//aux array to find the positions of (a-normM)/a <= 1e-6
	long int *b_pos = (long int *) malloc (image_size * sizeof(long int));	   	//valid positions of normM that meet (a-normM)/a <= 1e-6
	double *v = (double *) malloc (bands * sizeof(double));						//used to update normM in every iteration
	double *fvAux;                                                           	//float auxiliary array 
    long int J[endmembers];                                                 	//selected endmembers positions in input image

	if(image_size > bands){
		fvAux = (double *) malloc (image_size * sizeof(double));                
	}
	else{
		fvAux = (double *) malloc (bands * sizeof(double));
	}

    Load_Image(argv[1], image, cols, rows, bands, datatype);
    
   	/* printf("First 10 image:\n");
    for(i = 0; i < 10; i++){
        printf("%.15f\n", image[i]);
    } */
	/**************************** #END# - Load Image and allocate memory*******************************/

	gettimeofday(&t0,NULL);

    /**************************** #INIT# - Normalize image****************************************/
    if (normalize == 1){
        normalize_img(image, image_size, bands);   
    }

    for(i = 0; i < image_size * bands; i++){
        normM[i % image_size] += pow(image[i], 2); 
    }
	/**************************** #END# - Normalize image****************************************/

	max_val = max_Val(normM, image_size);

    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
	//if i == 1, normM1 = normM; 
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}

	i = 0;
	//while i <= r && max(normM)/nM > 1e-9
	while(i < endmembers && max_Val(normM, image_size)/max_val > 1e-9 ){
		//[a,b] = max(normM);
		a = max_Val(normM, image_size);

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
		
		//U(:,i) = M(:,b); 
		for(j = 0 ; j < bands; j++){
			U[i + endmembers*j] = image[image_size*j + J[i]];
		}
		
		//U(:,i) = U(:,i) - U(:,j)*(U(:,j)'*U(:,i));
		for(j = 0; j < i; j++){
			faux = 0;
			//(U(:,j)'*U(:,i))
			for(k = 0; k < bands; k++){
				faux += U[j + k*endmembers] * U[i + k*endmembers];
			}
			
			for(k = 0; k < bands; k ++){
				fvAux[k] = U[j + k*endmembers] * faux;
			}
			
			for(k = 0; k < bands; k ++){
				U[i + k*endmembers] = U[i + k*endmembers] - fvAux[k];
			}
					
		}
		
		//U(:,i) = U(:,i)/norm(U(:,i)); 
		faux = 0;
		for(j = 0; j < bands; j++){
			faux += pow(U[i + j*endmembers],2);
		}
		faux = sqrt(faux);
		for(j = 0; j < bands; j++){
			U[i + j*endmembers] = U[i + j*endmembers]/faux;
		}
		
		//v = U(:,i);
		for(j = 0; j < bands; j++){
			v[j] = U[i + j*endmembers];
		}

		// for j = i-1 : -1 : 1
		for(j = i - 1; j >= 0; j--){
			//(v'*U(:,j))
			faux = 0;
			for(k = 0; k < bands; k++){
				faux += v[k] * U[j + k*endmembers];
			}
			//(v'*U(:,j))*U(:,j);
			for(k = 0; k < bands; k ++){
				fvAux[k] = U[j + k*endmembers] * faux;
			}
			//v = v - (v'*U(:,j))*U(:,j);
			for(k = 0; k < bands; k ++){
				v[k] = v[k] - fvAux[k];
			}
		}

		//(v'*M).^2
		for(j = 0; j < image_size; j++){
			faux = 0;
			for(k = 0; k < bands; k++){
				 faux += v[k] * image[j + image_size*k];
			}
			fvAux[j] = powf(faux,2);
		}
		//normM = normM - (v'*M).^2;		
		for(j = 0; j < image_size; j++){
			normM[j] -= fvAux[j];
		}
			
		i = i + 1;
		
	}
	/**************************** #END# - FastSEPNMF algorithm*****************************************/
	
	gettimeofday(&tfin,NULL);
	t_sec  = (float)  (tfin.tv_sec - t0.tv_sec);
  	t_usec = (float)  (tfin.tv_usec - t0.tv_usec);
	secsFin = t_sec + t_usec/1.0e+6;

	printf("Endmembers:\n");
    for(i = 0; i < endmembers; i++){
        printf("%ld \t- %ld\n", i, J[i]);
    }

	printf("Total time:	\t%.5f segundos\n", secsFin);
    
    free(image);
    free(U);
    free(normM);
	free(normM1);
	free(normMAux);
	free(b_pos);
	free(fvAux);
	free(v);

    return 0;
}

long int max_val_extract_array(double *normMAux, long int *b_pos, long int b_pos_size){
	double max_val = -1;
	long int pos = -1;
	long int i;
    for (i = 0; i < b_pos_size; i++){
        if(normMAux[b_pos[i]] > max_val){
            max_val = normMAux[b_pos[i]];
			pos = i;
        }
    }
	return pos;
}


double max_Val(double *vector, long int image_size){
	double max_val = -1;
	long int i;

    for (i = 0; i < image_size; i++){
        if(vector[i] > max_val){
            max_val = vector[i];
        }
    }

	return max_val;
}


void normalize_img(double *image, long int image_size, int bands){
    long int i, j;
    long int row;

    double *D = (double *) calloc (image_size, sizeof(double));               //aux array to normalize the input image

    for(i = 0; i < image_size * bands; i++){
        D[i % image_size] += image[i]; 
    }

    for(i = 0; i < image_size; i++){
        D[i] = powf(D[i] + 1.0e-16, -1);
    }

    for (i = 0; i < bands; i++){
        row = i * image_size;
        for(j = 0; j < image_size; j++){
            image[row + j] = image[row + j] * D[j];
        } 
    }

    free(D);
}
