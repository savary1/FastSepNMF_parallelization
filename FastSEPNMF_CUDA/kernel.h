#ifndef _KERNEL_H

#define _KERNEL_H

#ifdef __cplusplus
extern "C"
#endif
//void normMactualization(float *v, float *image, int bands, float *normM, int image_size);
//void suma(float *v, float *out, int bands);
void act(float *v, float *image, int bands, float *normM, int image_size, int i, int rows);

#endif
