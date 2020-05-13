__kernel void update_normM(__global float* restrict cl_image, __global float* restrict cl_v, __global float* restrict cl_normM, int bands, int image_size) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int k;

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            faux += cl_v[k] * cl_image[k*image_size + id];
		}
        faux = faux * faux;
        cl_normM[id] -= faux;
    }    
}



__kernel void normalize_img(__global float* restrict cl_image, __global float* restrict cl_normM, int image_size, int bands) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int j;
    float imageNewVal, normAcum = 0, normAux = 0;

    if(id < image_size){
        for(j = 0; j < bands; j++){
            normAux += cl_image[j*image_size + id];
        }
    
        normAux = 1.0/(normAux + 1.0e-16);
        for(j = 0; j < bands; j++){
            imageNewVal = cl_image[j*image_size + id] * normAux;
            cl_image[j*image_size + id] = imageNewVal;
            normAcum += imageNewVal * imageNewVal;
        }
        cl_normM[id] = normAcum;
    }
}



__kernel void initialize_normM(__global float* restrict cl_image, __global float* restrict cl_normM, int image_size, int bands) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float normAcum = 0;
    int j;
    if(id < image_size){
        for(j = 0; j < bands; j++){
        normAcum += cl_image[j*image_size + id] * cl_image[j*image_size + id];
    }
    cl_normM[id] = normAcum;
    }    
}