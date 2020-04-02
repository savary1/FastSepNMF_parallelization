__kernel void update_normM(__global float* restrict clImage, __global float* restrict clV, __global float* restrict clNormM, const long int bands){
    unsigned int id = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

    clNormM[id] = clImage[id] + clV[id];
}