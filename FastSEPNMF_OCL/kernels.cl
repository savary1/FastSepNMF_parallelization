__kernel void update_normM(__global float* restrict clImage, __global float* restrict clV, __global float* restrict clNormM, int bands, int image_size) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int k;

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            faux += clV[k] * clImage[k*image_size + id];
		}
        faux = faux * faux;
        clNormM[id] -= faux;
    }    
}



__kernel void normalize_img(__global float* restrict clImage, __global float* restrict clNormM, int image_size, int bands) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int j;
    float imageNewVal, normAcum = 0, normAux = 0;

    if(id < image_size){
        for(j = 0; j < bands; j++){
            normAux += clImage[j*image_size + id];
        }
    
        normAux = 1.0/(normAux + 1.0e-16);
        for(j = 0; j < bands; j++){
            imageNewVal = clImage[j*image_size + id] * normAux;
            clImage[j*image_size + id] = imageNewVal;
            normAcum += imageNewVal * imageNewVal;
        }
        clNormM[id] = normAcum;
    }
}



__kernel void initializeNormM(__global float* restrict clImage, __global float* restrict clNormM, int image_size, int bands) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float normAcum = 0;
    int j;
    if(id < image_size){
        for(j = 0; j < bands; j++){
        normAcum += clImage[j*image_size + id] * clImage[j*image_size + id];
    }
    clNormM[id] = normAcum;
    }    
}