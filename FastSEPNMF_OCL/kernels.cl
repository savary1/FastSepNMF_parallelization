#pragma OPENCL EXTENSION cl_nv_pragma_unroll : enable

__kernel void update_normM(__global float* restrict clImage, __global float* restrict clV, __global float* restrict clNormM, int bands, int image_size){
    //int id = get_global_id(0);
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int k;

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            faux += clV[k] * clImage[id*bands + k];
		}
        faux = faux * faux;
        clNormM[id] -= faux;
    }    
}