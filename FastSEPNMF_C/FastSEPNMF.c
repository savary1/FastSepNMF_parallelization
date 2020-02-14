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
    long int i, *pos_max, b, j, b_pos_size;
    float max_val, a;

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
	float *normM1 = (float *) malloc (image_size, sizeof(float));			//copy of normM
	float *normMAux = (float *) malloc (image_size, sizeof(float));			//aux array to find the positions of (a-normM)/a <= 1e-6
	float *b_pos = (float *) malloc (image_size, sizeof(float));	        //valid positions of normM that meet (a-normM)/a <= 1e-6
    long int J[endmembers];                                                 //selected endmembers positions in input image

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
	
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}
	i = 0;

	while(i <= endmembers && max_Val(normM, image_size, pos_max)/max_val > 1e-9 ){
		a = max_Val(normM, image_size, pos_max);
		for(j = 0; j < image_size; j++){
			normMAux[j] = (a - normM)/a;
		}
		b_pos_size = 0;
		for(j = 0; j < image_size; j++){
			if (normMAux[j]<= 1.0e-6){
				b[b_pos_size] = j;
				b_pos_size++;
			}
		}
		
		
		
		
	}

    /**************************** #END# - FastSEPNMF algorithm*****************************************/

    printf("Maxval: %11.9f \n", max_val);
    printf("First 10 normM:\n");
    for(i = 0; i < 10; i++){
        printf("%9.6f\n", normM[i]);
    }
    

    free(image);
    free(U);
    free(normM);

    return 0;
}

int max_Val(float *vector, long int image_size, long int *pos_max){
	max_val = -1;
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
