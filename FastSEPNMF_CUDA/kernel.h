#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

void liberarMemoria(float *v_c, float *image_c, float *normM_c, float *image, float *normM, float *normM1_c, float *v, float *d_projections, int *d_index, float *h_projections, int *h_index); 
void selectDevice();
void reservarMemoria(int bands,long int image_size, float **v_c, float **image_c, float **normM_c, float **normM1_c, float **image, float **normM, float **v, float **d_projections, int **d_index, float **h_projections, int **h_index, int globalSize_reduction);
void actualizarNormM(float *v, int bands, float *normM, long int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c, float *t_act);
void copyNormM(float *normM, float *normM_c, long int image_size);
void normalizeImgC(float *image, long int image_size, int bands,float *image_c, int rows, float * normM_c, float *normM, float *normM1, float *normM1_c);
void calculateNormM(float *image, float *normM, float *normM1, long int image_size, int bands, int rows, float *image_c, float *normM_c, float *normM1_c);
void calculateMaxValExtract_2(int image_size, float *normM_c, float *normM1_c, float *d_projections, float *h_projections, int *d_index, int *h_index, float a);
void calculateMaxVal(int image_size, float *normM_c, float *d_projections, float *h_projections);


#ifdef __cplusplus
}
#endif


#endif

