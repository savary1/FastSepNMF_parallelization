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
    long int i;
    int h;
    float max_val;

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
    float *image = (float *) calloc (image_size * bands, sizeof(float));  //input image
    float *U = (float *) malloc (bands * endmembers * sizeof(float));       //selected endmembers
    float *normM = (float *) calloc (image_size, sizeof(float));           //normalized image
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



    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
    max_val = -1;
    for (i = 0; i < image_size; i++){
        if(normM[i] > max_val){
            max_val = normM[i];
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
