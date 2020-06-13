__kernel void normM_reduction(__global float* restrict cl_normM, int image_size, __global float* restrict cl_red_result){
    __local float l_red[2048];

    int l_size = get_local_size(0);
    unsigned int id = (get_group_id(0) * 2) * get_local_size(0) + get_local_id(0);
    unsigned int l_id = get_local_id(0);

    if(id < image_size){
        l_red[l_id] = cl_normM[id];
    }
    else{
        l_red[l_id] = -1000;
    }
    if(id + l_size < image_size){
        l_red[l_id + l_size] = cl_normM[id + l_size];
    }
    else{
        l_red[l_id + l_size] = -1000;
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = l_size; i > 0; i >>= 1) {
        if(l_id < i) {
            if(l_red[l_id] < l_red[l_id + i]){
                l_red[l_id] = l_red[l_id + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id == 0) {
        cl_red_result[get_group_id(0)] = l_red[0];
    }
}



__kernel void select_endmember(__global float* restrict cl_normM, __global float* restrict cl_normM1, int image_size, float a,__global float* restrict cl_red_result, __global int* restrict cl_red_result_pos){
    __local float l_red[2048];
    __local int l_red_pos[2048];
    __local int l_normM1[2048];

    int l_size = get_local_size(0);
    unsigned int id = (get_group_id(0) * 2) * get_local_size(0) + get_local_id(0);
    unsigned int l_id = get_local_id(0);

    if(id < image_size){
        l_red[l_id] =  (a - cl_normM[id]) / a;
        l_red_pos[l_id] = id;
        l_normM1[l_id] = cl_normM1[id];
    }
    else{
        l_red[l_id] = 1000;
    }
    if(id + l_size < image_size){
        l_red[l_id + l_size] = (a - cl_normM[id + l_size]) / a;
        l_red_pos[l_id + l_size] = id + l_size;
        l_normM1[l_id + l_size] = cl_normM1[id + l_size];
    }
    else{
        l_red[l_id + l_size] = 1000;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = l_size; i > 0; i >>= 1) {
        if(l_id < i) {
            if(l_red[l_id + i] <= 1.0e-6){
                if(l_red[l_id] > 1.0e-6 || (l_red[l_id] <= 1.0e-6 && l_normM1[l_id + i] > l_normM1[l_id])) {
                    l_red[l_id] = l_red[l_id + i];
                    l_red_pos[l_id] = l_red_pos[l_id + i];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(l_id == 0){
        if(l_red[0] <= 1.0e-6) {
            cl_red_result[get_group_id(0)] = l_normM1[l_id];
            cl_red_result_pos[get_group_id(0)] = l_red_pos[0];
        }
        else{
            cl_red_result[get_group_id(0)] = -1;
        }
    }
}



__kernel void update_normM(__global float* restrict cl_image, __global float* restrict cl_v, __global float* restrict cl_normM, int bands, int image_size) {
    __local float l_v[224];
    
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int k;

    if(get_local_size(0) < bands){
	    		for(k = get_local_id(0); k < bands; k += get_local_size(0)){
	      			l_v[k]=cl_v[k];
	    		}
	}
    else{
        if(get_local_id(0) < bands){
            l_v[get_local_id(0)] = cl_v[get_local_id(0)];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            faux += l_v[k] * cl_image[k*image_size + id];
		}
        faux = faux * faux;
        cl_normM[id] -= faux;
    }    
}



__kernel void normalize_img(__global float* restrict cl_image, __global float* restrict cl_normM, __global float* restrict cl_normM1, int image_size, int bands) {
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
        cl_normM1[id] = normAcum;
    }
}



__kernel void initialize_normM(__global float* restrict cl_image, __global float* restrict cl_normM, __global float* restrict cl_normM1, int image_size, int bands) {
    unsigned int id = get_group_id(0) * get_local_size(0) + get_local_id(0);
    float normAcum = 0;
    int j;
    if(id < image_size){
        for(j = 0; j < bands; j++){
            normAcum += cl_image[j*image_size + id] * cl_image[j*image_size + id];
        }
        cl_normM[id] = normAcum;
        cl_normM1[id] = normAcum;
    }        
}