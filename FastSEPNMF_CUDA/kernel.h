#ifndef KERNEL_H_
#define KERNEL_H_

#ifdef __cplusplus
extern "C" {
#endif

void liberarMemoria(float *v_c, float *image_c, float *normM_c, float *image, float *normM, float *v); 
void selectDevice();
void reservarMemoria(int bands,long int image_size, float **v_c, float **image_c, float **normM_c, float **image, float **normM, float **v);
void actualizarNormM(float *v, int bands, float *normM, long int image_size, int i, int rows, float *v_c, float *image_c, float *normM_c, float *t_act);
void copyNormM(float *normM, float *normM_c, long int image_size);
void normalizeImgC(float *image, long int image_size, int bands,float *image_c, int rows, float * normM_c, float *normM, float *normM1, float *t_copia_v, float *t_normalizar, float *t_copia_norm);
void calculateNormM(float *image, float *normM, float *normM1, long int image_size, int bands, int rows, float *image_c, float *normM_c);

#ifdef __cplusplus
}
#endif


#endif

