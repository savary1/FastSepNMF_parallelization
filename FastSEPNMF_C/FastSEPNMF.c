//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include "ReadWrite.h"

void normalize_img(float *image, long int image_size, int bands);

int main (int argc, char* argv[]){

    int rows, cols, bands; //size of the image
    int datatype;
    int endmembers;
    int normalize;
    long int i, *pos_max, j, b_pos_size, d, k;
    float max_val, a, b, faux;

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
    float *image = (float *) calloc (image_size * bands, sizeof(float));    //input image
    float *U = (float *) malloc (bands * endmembers * sizeof(float));       //selected endmembers
    float *normM = (float *) calloc (image_size, sizeof(float));            //normalized image
	float *normM1 = (float *) malloc (image_size * sizeof(float));			//copy of normM
	float *normMAux = (float *) malloc (image_size * sizeof(float));		//aux array to find the positions of (a-normM)/a <= 1e-6
	float *b_pos = (float *) malloc (image_size * sizeof(float));	        //valid positions of normM that meet (a-normM)/a <= 1e-6
	float *fvAux;                                                           // float auxiliary array
	float *v = (float *) malloc (bands * sizeof(float)); 
    long int J[endmembers];                                                 //selected endmembers positions in input image

	if(image_size > bands){
		fvAux = (float *) malloc (image_size * sizeof(float));                
	}
	else{
		fvAux = (float *) malloc (bands * sizeof(float));
	}

    Load_Image(argv[1], image, cols, rows, bands, datatype);
    
    printf("First 10 image:\n");
    for(i = 0; i < 10; i++){
        printf("%f\n", image[i]);
    }

    printf("First 10 image:\n");
    for(i = 350*350; i < 350*350 + 10; i++){
        printf("%f\n", image[i]);
    }

    /**************************** #END# - Load Image and allocate memory*******************************/



    /**************************** #INIT# - Normalize image****************************************/

    if (normalize == 1){
        normalize_img(image, image_size, bands);   
    }

    for(i = 0; i < image_size * bands; i++){
        normM[i % image_size] += powf(image[i], 2); 
    }

    /**************************** #END# - Normalize image****************************************/

	max_val = max_Val(normM, image_size, pos_max);

    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
	//if i == 1, normM1 = normM; 
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}
	i = 0;

	while(i <= endmembers && max_Val(normM, image_size, pos_max)/max_val > 1e-9 ){
		//[a,b] = max(normM);
		a = max_Val(normM, image_size, pos_max);
		//(a-normM)/a
		for(j = 0; j < image_size; j++){
			normMAux[j] = (a - normM)/a;
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
			d = max_val_extract_array(normMAux, b_pos, b_pos_size);
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
				faux += U[j + k*endmembers] * U[i + k*endmembers]; // hacerlo con un array para paralelizar
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
			faux += powf(U[i + j*endmembers],2);
		}
		faux = sqrtf(faux);
		
		for(j = 0; j < bands; j++){
			U[i + j*endmembers] = U[i + j*endmembers]/faux;
		}
		
		//v = U(:,i);
		for(j = 0; j < bands; j++){
			v[j] = U[i + j*endmembers];
		}
		
		for(j = i - 1; j >= 0; j--){
			//(v'*U(:,j))
			faux = 0;
			for(k = 0; k < bands; k++){
				faux += v[k] * U[j + k*endmembers]; // hacerlo con un array para paralelizar
			}
			for(k = 0; k < bands; k ++){
				fvAux[k] = U[j + k*endmembers] * faux;
			}
			
			for(k = 0; k < bands; k ++){
				v[k] = v[k] - fvAux[k];
			}
		}
		
		//(v'*M).^2
		for(j = 0; j < image_size; j++){
			faux = 0;
			for(k = 0; k < bands; k++){
				 faux = v[k] * M[j + bands*k];
			}
			fvAux[j] = powf(faux,2);
		}
		
		for(j = 0; j < image_size; j++){
			normM[j] -= fvAux[j];
		}
		
		i = i + 1;
		
	}

    /**************************** #END# - FastSEPNMF algorithm*****************************************/

    printf("Maxval: %11.9f \n", max_val);
    printf("First 10 normM:\n");
    for(i = 0; i < 10; i++){
        printf("%9.6f\n", normM[i]);
    }
	
	printf("Endmembers:\n");
    for(i = 0; i < endmembers; i++){
        printf("%ld\n", J[i]);
    }
    

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

long int max_val_extract_array(float *normMAux, float *b_pos, long int b_pos_size){
	float max_val = -1;
	long int pos = -1;
    for (i = 0; i < b_pos_size; i++){
        if(normMAux[b_pos[i]] > max_val){
            max_val = vector[i];
			pos = i;
        }
    }
	return pos;
	
}


float max_Val(float *vector, long int image_size, long int *pos_max){
	float max_val = -1;
    for (i = 0; i < image_size; i++){
        if(vector[i] > max_val){
            max_val = vector[i];
			pos_max = i;
        }
    }
	return max_val;
}


void normalize_img(float *image, long int image_size, int bands){
    long int i, j;
    long int row;

    float *D = (float *) calloc (image_size, sizeof(float));               //aux array for normalize the input image

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
