//FastSEPNMF AROA AYUSO   DAVID SAVARY   2020

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdlib.h>
#include "ReadWrite.h"

void normalize_img(float *image, long int image_size, int bands);
long int max_val_extract_array(float *normMAux, long int *b_pos, long int b_pos_size);
float max_Val(float *vector, long int image_size);

int main (int argc, char* argv[]){

    int rows, cols, bands; //size of the image
    int datatype;
    int endmembers;
    int normalize;
    long int i, j, b_pos_size, d, k;
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
	long int *b_pos = (long int *) malloc (image_size * sizeof(long int));	        //valid positions of normM that meet (a-normM)/a <= 1e-6
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
    
   /* printf("First 10 image:\n");
    for(i = 0; i < 10; i++){
        printf("%f\n", image[i]);
    }

    printf("First 10 image:\n");
    for(i = 350*350; i < 350*350 + 10; i++){
        printf("%f\n", image[i]);
    }*/

    /**************************** #END# - Load Image and allocate memory*******************************/



    /**************************** #INIT# - Normalize image****************************************/

    if (normalize == 1){
        normalize_img(image, image_size, bands);   
    }

    for(i = 0; i < image_size * bands; i++){
        normM[i % image_size] += powf(image[i], 2); 
    }

    /**************************** #END# - Normalize image****************************************/

	max_val = max_Val(normM, image_size);

    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
	//if i == 1, normM1 = normM; 
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}
	i = 0;
	printf("Holaaaa -1\n");
	while(i < endmembers && max_Val(normM, image_size)/max_val > 1e-9 ){
		printf("Holaaaa 0\n")
		//[a,b] = max(normM);
		a = max_Val(normM, image_size);
		printf("El valor de a es :  %f\n", a);
		printf("Holaaaa 1\n");
		//(a-normM)/a
		for(j = 0; j < image_size; j++){
			normMAux[j] = (a - normM[j])/a;
		}
		/////////////////////////// debug
		printf("Los primeros 10 valores de nomMAux son:");
		for (j = 0; j < 10; j++){
			printf("%f\n", normMAux[j]);
		}
		///////////////////////  fin debug
		
		printf("Holaaaa 2\n");
		//b = find((a-normM)/a <= 1e-6); 
		b_pos_size = 0;
		for(j = 0; j < image_size; j++){
			if (normMAux[j]<= 1.0e-6){
				b_pos[b_pos_size] = j;
				b_pos_size++;
			}
		}
		
		//////////////////////////////////////    -   For de debug
		printf("array b\n");
		
		for(j = 0; j < b_pos_size; j ++){
			printf("%ld\n", b_pos[j]);
		}
		printf("b_pos_size :  %ld\n", b_pos_size);
		
		//////////////////////////////////// - Fin for debug	
		
		//if length(b) > 1, [c,d] = max(normM1(b)); b = b(d);
		if (b_pos_size > 1){
			d = max_val_extract_array(normM1, b_pos, b_pos_size);
			b = b_pos[d];
			J[i] = b;
		}
		else{ // comprobar si siempre tiene valores b_pos
			J[i] = b_pos[0];
		}
		
		printf("Holaaaa 3\n");
		
		//U(:,i) = M(:,b); 
		for(j = 0 ; j < bands; j++){
			U[i + endmembers*j] = image[image_size*j + J[i]];
		}
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de U: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", U[i + endmembers*j]); 	// son correctos en i = 0
		}
		
		//////////////////////////////////// - Fin debug
		
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
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de U: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", U[i + endmembers*j]); 	
		}
		
		//////////////////////////////////// - Fin debug
		printf("Holaaaa 4\n");
		
		//U(:,i) = U(:,i)/norm(U(:,i)); 
		faux = 0;
		for(j = 0; j < bands; j++){
			faux += powf(U[i + j*endmembers],2);
		}
		faux = sqrtf(faux);
		
		printf("el valor de norm(U(:,i)) es: %f\n ", faux);
		
		for(j = 0; j < bands; j++){
			U[i + j*endmembers] = U[i + j*endmembers]/faux;
		}
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de U: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", U[i + endmembers*j]); 	
		}
		
		//////////////////////////////////// - Fin debug
		
		//v = U(:,i);
		for(j = 0; j < bands; j++){
			v[j] = U[i + j*endmembers];
		}
		printf("Holaaaa 5\n");
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
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de v: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", v[j]); 	
		}
		
		//////////////////////////////////// - Fin for debug
		printf("Holaaaa 6\n");
		//(v'*M).^2
		for(j = 0; j < image_size; j++){
			faux = 0;
			for(k = 0; k < bands; k++){
				 faux += v[k] * image[j + image_size*k];
			}
			fvAux[j] = powf(faux,2);
		}
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de (v'*M).^2: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", fvAux[j]); 	
		}
		
		//////////////////////////////////// - Fin for debug
		
		for(j = 0; j < image_size; j++){
			normM[j] -= fvAux[j];
		}
		
		//////////////////////////////////////   debug
		printf("Los primeros valores de normM: \n");
		for(j = 0; j < 10; j ++){
			printf("%f\n", normM[j]); 	
		}
		
		//////////////////////////////////// - Fin for debug
			
		i = i + 1;
		
	}

    /**************************** #END# - FastSEPNMF algorithm*****************************************/
	/*printf("Maxval: %11.9f \n", max_val);
    printf("First 10 normM:\n");
    for(i = 0; i < 10; i++){
        printf("%9.6f\n", normM[i]);
    }*/
	
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

long int max_val_extract_array(float *normMAux, long int *b_pos, long int b_pos_size){
	float max_val = -1;
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


float max_Val(float *vector, long int image_size){
	float max_val = -1;
	long int i;
	printf("Buenas tardes 0\n");
    for (i = 0; i < image_size; i++){
        if(vector[i] > max_val){
            max_val = vector[i];
        }
    }
	printf("La pos de max val es: %ld\n", pos);
	printf("Buenas tardes 1\n");
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
