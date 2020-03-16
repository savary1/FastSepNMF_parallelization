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

	struct timeval t0, tfin, t1, t2;
	float secsFin, t_sec, t_usec, tNorm, tFindMax, tb1, tb2, tb3, tb4, tb5, tb6, tb7, tb8, tb9, tb10, tb11, tb12, tb13, tb14;
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
	/**************************** #END# - Load Image and allocate memory*******************************/

	gettimeofday(&t0,NULL);

    /**************************** #INIT# - Normalize image****************************************/
    if (normalize == 1){
        normalize_img(image, image_size, bands);   
    }

	gettimeofday(&t1,NULL);
	//Este for se puede separa en 2, el de fuera de longitud image size y el de dentro vectorizarlo
    for(i = 0; i < image_size * bands; i++){
        normM[i % image_size] += image[i] * image[i]; 
    }
	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
  	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	tNorm = t_sec + t_usec/1.0e+6;

	/**************************** #END# - Normalize image****************************************/
	gettimeofday(&t1,NULL);
	max_val = max_Val(normM, image_size);
	gettimeofday(&t2,NULL);
	t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
  	t_usec = (float)  (t2.tv_usec - t1.tv_usec);
	tFindMax = t_sec + t_usec/1.0e+6;
    /**************************** #INIT# - FastSEPNMF algorithm****************************************/
	//if i == 1, normM1 = normM; 
	for(i = 0; i < image_size; i++){
		normM1[i] = normM[i];
	}

	tb1 = tb2 = tb3 = tb4 = tb5 = tb6 = tb7 = tb8 = tb9 = tb10 = tb11 = tb12 = tb13 = tb14 = 0;

	i = 0;
	//while i <= r && max(normM)/nM > 1e-9
	while(i < endmembers && max_Val(normM, image_size)/max_val > 1e-9 ){
		//[a,b] = max(normM);
		gettimeofday(&t1,NULL);
		a = max_Val(normM, image_size);
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tFindMax = tFindMax + t_sec + t_usec/1.0e+6;

		//(a-normM)/a
		gettimeofday(&t1,NULL);
		for(j = 0; j < image_size; j++){
			normMAux[j] = (a - normM[j])/a;
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb1 = tb1 + t_sec + t_usec/1.0e+6;

		//b = find((a-normM)/a <= 1e-6);
		gettimeofday(&t1,NULL);
		b_pos_size = 0;
		for(j = 0; j < image_size; j++){
			if (normMAux[j]<= 1.0e-6){
				b_pos[b_pos_size] = j;
				b_pos_size++;
			}
		}	
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb2 = tb2 + t_sec + t_usec/1.0e+6;
		
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
		gettimeofday(&t1,NULL); //MIRAR SI SE PUEDEN HACER LOS ACCESOS A MEMORIA ADYACENTES
		for(j = 0 ; j < bands; j++){
			U[i + endmembers*j] = image[image_size*j + J[i]];
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb3 = tb3 + t_sec + t_usec/1.0e+6;
		
		//U(:,i) = U(:,i) - U(:,j)*(U(:,j)'*U(:,i));
		for(j = 0; j < i; j++){
			faux = 0;
			//(U(:,j)'*U(:,i))
			gettimeofday(&t1,NULL);
			for(k = 0; k < bands; k++){//MIRAR SI LOS ACCESOS DE PUEDEN HACER ADYACENTES
				faux += U[j + k*endmembers] * U[i + k*endmembers];
			}
			gettimeofday(&t2,NULL);
			t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
			t_usec = (float)  (t2.tv_usec - t1.tv_usec);
			tb4 = tb4 + t_sec + t_usec/1.0e+6;
			
			gettimeofday(&t1,NULL);//MIRAR SI LOS ACCESOS A MEMORIA SE PUEDEN HACER ADYACENTES
			for(k = 0; k < bands; k ++){
				fvAux[k] = U[j + k*endmembers] * faux;
			}
			gettimeofday(&t2,NULL);
			t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
			t_usec = (float)  (t2.tv_usec - t1.tv_usec);
			tb5 = tb5 + t_sec + t_usec/1.0e+6;
			
			gettimeofday(&t1,NULL);
			for(k = 0; k < bands; k ++){//ESTE NO LO VECTORIZA Y CREO QUE SI SE PUEDE. INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
				U[i + k*endmembers] = U[i + k*endmembers] - fvAux[k];
			}
			gettimeofday(&t2,NULL);
			t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
			t_usec = (float)  (t2.tv_usec - t1.tv_usec);
			tb6 = tb6 + t_sec + t_usec/1.0e+6;
					
		}
		
		//U(:,i) = U(:,i)/norm(U(:,i));
		gettimeofday(&t1,NULL);
		faux = 0;
		for(j = 0; j < bands; j++){//INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
			faux += U[i + j*endmembers] * U[i + j*endmembers];
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb7 = tb7 + t_sec + t_usec/1.0e+6;
		faux = sqrt(faux);
		gettimeofday(&t1,NULL);
		for(j = 0; j < bands; j++){	//NO LO VECTORIZA, CREO QUE SI SE PUEDE. INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
			U[i + j*endmembers] = U[i + j*endmembers]/faux;
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb8 = tb8 + t_sec + t_usec/1.0e+6;
		
		//v = U(:,i);
		gettimeofday(&t1,NULL);
		for(j = 0; j < bands; j++){//INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
			v[j] = U[i + j*endmembers];
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb9 = tb9 + t_sec + t_usec/1.0e+6;

		// for j = i-1 : -1 : 1
		gettimeofday(&t1,NULL);
		for(j = i - 1; j >= 0; j--){
			//(v'*U(:,j))
			faux = 0;
			for(k = 0; k < bands; k++){//INTENTAR HACER ACCESOS A MEMORIA ADYACENTES
				faux += v[k] * U[j + k*endmembers];
			}
			//(v'*U(:,j))*U(:,j);//HACER ACCESOS A MEMORIA ADYACENTES
			for(k = 0; k < bands; k ++){
				fvAux[k] = U[j + k*endmembers] * faux;
			}
			//v = v - (v'*U(:,j))*U(:,j);
			for(k = 0; k < bands; k ++){//DICE QUE NO LO HACE PORQUE PARECE INEFICIENTE
				v[k] = v[k] - fvAux[k];
			}
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb10 = tb10 + t_sec + t_usec/1.0e+6;

		//(v'*M).^2
		gettimeofday(&t1,NULL);
		for(j = 0; j < image_size; j++){
			faux = 0;
			for(k = 0; k < bands; k++){//INTENTAR HACER ACCESOS ADYACENTES
				faux += v[k] * image[j + image_size*k];
			}
			fvAux[j] = faux * faux;
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb13 = tb13 + t_sec + t_usec/1.0e+6;
		//normM = normM - (v'*M).^2;
		gettimeofday(&t1,NULL);		
		for(j = 0; j < image_size; j++){//DICE QUE PARECE INEFICIENTE VECTORIZAR
			normM[j] -= fvAux[j];
		}
		gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		tb14 = tb14 + t_sec + t_usec/1.0e+6;
			
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
	printf("T norm:	\t%.5f segundos\n", tNorm);
	printf("T find max val:	\t%.5f segundos\n", tFindMax);
	printf("T 1:	\t%.5f segundos\n", tb1);
	printf("T 2:	\t%.5f segundos\n", tb2);
	printf("T 3:	\t%.5f segundos\n", tb3);
	printf("T 4:	\t%.5f segundos\n", tb4);
	printf("T 5:	\t%.5f segundos\n", tb5);
	printf("T 6:	\t%.5f segundos\n", tb6);
	printf("T 7:	\t%.5f segundos\n", tb7);
	printf("T 8:	\t%.5f segundos\n", tb8);
	printf("T 9:	\t%.5f segundos\n", tb9);
	printf("T 10:	\t%.5f segundos\n", tb10);
	printf("T 11:	\t%.5f segundos\n", tb11);
	printf("T 12:	\t%.5f segundos\n", tb12);
	printf("T 13:	\t%.5f segundos\n", tb13);
	printf("T 14:	\t%.5f segundos\n", tb14);

    
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
    for (i = 0; i < b_pos_size; i++){ //DICE QUE NO SE PUEDE VECTORIZAR PORQUE MAXVAL DEPENDE DEL MAXVAL DE LA ITERACIÓN ANTERIOR
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
    long int i, j, k;
    long int row;

	struct timeval t1, t2;
	float secsFin, t_sec, t_usec, tb1, tb2, tb3;

    double *D = (double *) calloc (image_size, sizeof(double));               //aux array to normalize the input image

    for(i = 0; i < bands; i++){
		row = i * image_size;
		for(k = 0; k < image_size; k++){
        	D[k] += image[row + k];
		} 
    }

    for(i = 0; i < image_size; i++){
        D[i] = powf(D[i] + 1.0e-16, -1);
    }

	//Esto se puede hacer en un for de longitud Bands*imagesize
	gettimeofday(&t1,NULL);
	#pragma omp for
    for (i = 0; i < bands * image_size; i++){
            image[i] = image[i] * D[i % image_size];
    }
	gettimeofday(&t2,NULL);
		t_sec  = (float)  (t2.tv_sec - t1.tv_sec);
		t_usec = (float)  (t2.tv_usec - t1.tv_usec);
		printf("T3 norm: %f\n",t_sec + t_usec/1.0e+6);

    free(D);
}
