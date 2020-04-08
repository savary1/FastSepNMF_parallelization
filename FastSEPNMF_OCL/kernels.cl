__kernel void update_normM(__global float* restrict clImage, __global float* restrict clV, __global float* restrict clNormM, int bands, int image_size){
    int id = get_global_id(0);
    int k;

    float faux = 0;
    if(id < image_size){
        for(k = 0; k < bands; k++){
            faux += clV[k] * clImage[id*bands + k];
            //faux += clV[k];
            // faux += 1;
		}
        faux = faux * faux;
        clNormM[id] -= faux;
        // clNormM[id] = clV[69];
    }
    // for(j = 0; j < image_size; j++){
		// 	faux = 0;
		// 	for(k = 0; k < bands; k++){//INTENTAR HACER ACCESOS ADYACENTES
		// 		faux += v[k] * image[j*bands + k];
		// 	}
		// 	fvAux[j] = faux * faux;
		// 	normM[j] -= fvAux[j];
		// }
    
}