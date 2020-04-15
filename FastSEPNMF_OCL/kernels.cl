#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

__kernel void update_normM(__global float* restrict clImage, __global float* restrict clV, __global float* restrict clNormM, int bands, int image_size){
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int k;

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            //faux += clV[k] * clImage[id*bands + k];
            faux += clV[k] * clImage[k*image_size + id];
		}
        faux = faux * faux;
        clNormM[id] -= faux;
        //clNormM[id] = clImage[id];
    }    
}



__kernel void normalize_img(__global float* restrict clImage, __global float* restrict clImageHost, __global float* restrict clNormM, int image_size, int bands){
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
            // clImageHost[id*bands + j] = imageNewVal;
            //clImageHost[j*image_size + id] = imageNewVal;
            normAcum += imageNewVal * imageNewVal;
        }
        clNormM[id] = normAcum;
    }
}