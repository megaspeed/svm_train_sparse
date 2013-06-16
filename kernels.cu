#include <cuda_runtime_api.h>
#include <float.h>
#include "svm_data.h"
#include "device_launch_parameters.h"
/**
* Set initial values of the obj functions and alphas
* @param d_a device pointer to the array with the alphas
* @param d_f device pointer to the intermediate values of f 
* @param d_y device pointer to the array with binary labels
* @param ntraining number of training samples in the training set
*/
__global__ void initialization( float *d_a, float *d_f, int *d_y, int ntraining, int ntasks)
{
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
	while ( i < ntraining && j < ntasks)
	{
		d_a[i+j*ntraining] = 0.;
		d_f[i+j*ntraining] = -1.*d_y[i];
		i += blockDim.x*gridDim.x;
	}
	__syncthreads();
}
/**
* Make local reduce of treshold parameter Bup
* @param d_y device pointer to the array with binary labels
* @param d_a device pointer to the array with the alphas
* @param d_f device pointer to the intermediate values of f 
* @param d_bup_local device pointer to localy reduced Bup
* @param d_Iup_local device pointer to the array Bup indeces
* @param d_C parameter C
* @param ntraining # of training samples
*/
__global__ void Local_Reduce_Min(int* d_y, float* d_a, float *d_f, float *d_bup_local,
								 unsigned int* d_Iup_local, float *d_C, int ntraining)
{
	extern __shared__ float reducearray[];
	unsigned int tid = threadIdx.x;
	unsigned int blocksize = blockDim.x;
	unsigned int gridsize = blocksize*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	float *minreduction = (float*)reducearray;
	unsigned int *minreductionid = (unsigned int*)&reducearray[blocksize];
	minreduction[tid] = (float)FLT_MAX;
	minreductionid[tid] = i;
	float alpha_i;
	int y_i;
	while ( i < ntraining)
	{
		alpha_i = d_a[i];
		y_i = d_y[i];
		if ((   (y_i==1  && alpha_i>0 && alpha_i<*d_C) ||
			(y_i==-1 && alpha_i>0 && alpha_i<*d_C)) ||
			(y_i==1  && alpha_i==0)|| (y_i==-1 && alpha_i==*d_C))
		{
			if (minreduction[tid] > d_f[i])
			{
				minreduction[tid] = d_f[i];
				minreductionid[tid] = i;
			}
		}
		i += gridsize;
	}
	__syncthreads();
	if (blocksize >= 512){if(tid < 256){if(minreduction[tid] >  minreduction[tid+256])
	{  minreduction[tid] =   minreduction[tid+256];
	minreductionid[tid] = minreductionid[tid+256];}}
	__syncthreads();}
	if (blocksize >= 256){if(tid < 128){if(minreduction[tid] >  minreduction[tid+128])
	{  minreduction[tid] =   minreduction[tid+128];
	minreductionid[tid] = minreductionid[tid+128];}}
	__syncthreads();}
	if (blocksize >= 128){if(tid < 64){if(minreduction[tid] >  minreduction[tid+64])
	{ minreduction[tid] =   minreduction[tid+64];
	minreductionid[tid] = minreductionid[tid+64];}}
	__syncthreads();}

	if (tid < 32){	if(blocksize >= 64){if(minreduction[tid] >  minreduction[tid+32])
	{  minreduction[tid] =   minreduction[tid+32];
	minreductionid[tid] = minreductionid[tid+32];}}
	if(blocksize >= 32){if(minreduction[tid] >  minreduction[tid+16])
	{  minreduction[tid] =   minreduction[tid+16];
	minreductionid[tid] = minreductionid[tid+16];}}
	if(blocksize >= 16){if(minreduction[tid] >  minreduction[tid+ 8])
	{  minreduction[tid] =   minreduction[tid+ 8];
	minreductionid[tid] = minreductionid[tid+ 8];}}
	if(blocksize >= 8){if( minreduction[tid] >  minreduction[tid+ 4])
	{  minreduction[tid] =   minreduction[tid+ 4];
	minreductionid[tid] = minreductionid[tid+ 4];}}
	if(blocksize >= 4){if( minreduction[tid] >  minreduction[tid+ 2])
	{  minreduction[tid] =   minreduction[tid+ 2];
	minreductionid[tid] = minreductionid[tid+ 2];}}
	if(blocksize >= 2){if( minreduction[tid] >  minreduction[tid+ 1])
	{  minreduction[tid] =   minreduction[tid+ 1];
	minreductionid[tid] = minreductionid[tid+ 1];}}}

	if (tid == 0)
	{
		d_bup_local[blockIdx.x] = minreduction[tid];
		d_Iup_local[blockIdx.x] = minreductionid[tid];
	}
}

/**
* Make local reduce of treshold parameter Blow
* @param d_y device pointer to the array with binary labels
* @param d_a device pointer to the array with the alphas
* @param d_f device pointer to the intermediate values of f 
* @param d_blow_local device pointer to localy reduced Blow
* @param d_Ilow_local device pointer to the array Bup indeces
* @param d_C parameter C
* @param ntraining # of training samples
*/
__global__ void Local_Reduce_Max(int* d_y, float* d_a, float *d_f, float *d_blow_local,
								 unsigned int* d_Ilow_local, float *d_C, int ntraining)
{
	extern __shared__ float reducearray[];
	unsigned int tid = threadIdx.x;
	unsigned int blocksize = blockDim.x;
	unsigned int gridsize = blocksize*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	float *maxreduction = (float*)reducearray;
	int *maxreductionid = (int*)&reducearray[blocksize];
	maxreduction[tid] = -1.*(float)FLT_MAX;	
	maxreductionid[tid] = i;
	float alpha_i;
	int y_i;
	while ( i < ntraining)
	{
		alpha_i = d_a[i];
		y_i = d_y[i];
		if ((   (y_i==1  && alpha_i>0 && alpha_i<*d_C) ||
			(y_i==-1 && alpha_i>0 && alpha_i<*d_C)) ||
			(y_i==1  && alpha_i==*d_C)|| (y_i==-1 && alpha_i==0))
		{
			if (maxreduction[tid] < d_f[i])
			{
				maxreduction[tid] = d_f[i];
				maxreductionid[tid] = i;
			}
		}
		i += gridsize;
	}
	__syncthreads();
	if (blocksize >= 512){if(tid < 256){if(maxreduction[tid] <  maxreduction[tid+256])
	{  maxreduction[tid] =   maxreduction[tid+256];
	maxreductionid[tid] = maxreductionid[tid+256];}}
	__syncthreads();}
	if (blocksize >= 256){if(tid < 128){if(maxreduction[tid] <  maxreduction[tid+128])
	{  maxreduction[tid] =   maxreduction[tid+128];
	maxreductionid[tid] = maxreductionid[tid+128];}}
	__syncthreads();}
	if (blocksize >= 128){if(tid < 64){if(maxreduction[tid] <  maxreduction[tid+64])
	{ maxreduction[tid] =  maxreduction[tid+64];
	maxreductionid[tid] = maxreductionid[tid+64];}}
	__syncthreads();}

	if (tid < 32){	if(blocksize >= 64){if(maxreduction[tid] <  maxreduction[tid+32])
	{  maxreduction[tid] =   maxreduction[tid+32];
	maxreductionid[tid] = maxreductionid[tid+32];}}
	if(blocksize >= 32){if(maxreduction[tid] <  maxreduction[tid+16])
	{  maxreduction[tid] =   maxreduction[tid+16];
	maxreductionid[tid] = maxreductionid[tid+16];}}
	if(blocksize >= 16){if(maxreduction[tid] <  maxreduction[tid+ 8])
	{  maxreduction[tid] =   maxreduction[tid+ 8];
	maxreductionid[tid] = maxreductionid[tid+ 8];}}
	if(blocksize >= 8){if( maxreduction[tid] <  maxreduction[tid+ 4])
	{  maxreduction[tid] =   maxreduction[tid+ 4];
	maxreductionid[tid] = maxreductionid[tid+ 4];}}
	if(blocksize >= 4){if( maxreduction[tid] <  maxreduction[tid+ 2])
	{  maxreduction[tid] =   maxreduction[tid+ 2];
	maxreductionid[tid] = maxreductionid[tid+ 2];}}
	if(blocksize >= 2){if( maxreduction[tid] <  maxreduction[tid+ 1])
	{  maxreduction[tid] =   maxreduction[tid+ 1];
	maxreductionid[tid] = maxreductionid[tid+ 1];}}}

	if (tid == 0)
	{
		d_blow_local[blockIdx.x] = maxreduction[tid];
		d_Ilow_local[blockIdx.x] = maxreductionid[tid];
	}
}

/**
* Recalculate value of f function
* @param d_f device pointer to the intermediate values of f 
* @param d_k device pointer to the kernel matrix
* @param d_y device pointer to the array with binary labels
* @param d_delta_a device pointer to the array with  alphas_new-alphas_old
* @param d_I_global device pointer to global index of kernel matrix row
* @param d_I_cache device pointer to index of kernel matrix row in cache
* @param d_active device pointer to the array tasks statuses
* @param ntraining # of training samples
*/
__global__ void Map( float *d_f, float *d_k, int *d_y, float *d_delta_a, unsigned int* d_I_global,
					unsigned int *d_I_cache, float *params, int *d_active, int ntraining)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;

	if (d_active[j] == 1)
	{
		while ( i < ntraining)
		{
			d_f[j*ntraining+i] += d_delta_a[2*j]*d_y[d_I_global[2*j]]*exp(-params[2*j+1]*d_k[d_I_cache[2*j]*ntraining+i]) +  /*up */
				d_delta_a[2*j+1]*d_y[d_I_global[2*j+1]]*exp(-params[2*j+1]*d_k[d_I_cache[2*j+1]*ntraining+i]);   /*low*/
			i += gridsize;
		}
	}
}
/**
* Update curently optimazed Lagrange multiplyers
* @param d_k device pointer to the kernel matrix
* @param d_y device pointer to the array with binary labels
* @param d_f device pointer to the intermediate values of f 
* @param d_a device pointer to the array with Lagrange multiplyers alphas
* @param d_I_global device pointer to global index of kernel matrix row
* @param d_I_cache device pointer to index of kernel matrix row in cache
* @param d_C parameter C
* @param d_active device pointer to the array tasks statuses
* @param ntraining # of training samples
*/
__global__ void Update(float *d_k, int *d_y, float *d_f, float *d_a, float *d_delta_a, 
					   unsigned int *d_I_global, unsigned int *d_I_cache, float* C, int *d_active, int ntraining)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//task number
	if (d_active[i] == 1)
	{
		int g_Iup = d_I_global[2*i];
		int g_Ilow = d_I_global[2*i+1];
		float alpha_up_old =d_a[i*ntraining+g_Iup];
		float alpha_low_old =d_a[i*ntraining+g_Ilow];

		float alpha_up_new = max(0, min(alpha_up_old + 
			(d_y[g_Iup]*(d_f[i*ntraining+g_Ilow]-d_f[i*ntraining+g_Iup])/
			(2- 2*exp(-C[2*i+1]*d_k[d_I_cache[2*i+1]*ntraining+g_Iup]))), C[2*i]));

		float alpha_low_new = max(0, min(alpha_low_old+
			d_y[g_Iup]*d_y[g_Ilow]*(alpha_up_old-alpha_up_new), C[2*i]));

		d_delta_a[2*i] = alpha_up_new-alpha_up_old;
		d_delta_a[2*i+1] = alpha_low_new-alpha_low_old;
		d_a[i*ntraining+g_Iup] = alpha_up_new;
		d_a[i*ntraining+g_Ilow] = alpha_low_new;
	}
	__syncthreads();
}
__device__ float get_selfdot(float *xv, int *xi, int *iax, unsigned int irow )
{
	float res = 0;
	for (int i = iax[irow]; i < iax[irow+1]; i++)
		res += xv[i]*xv[i];
	return res;
}
__device__ float makedot(float *xv, int *xi, int *iax, unsigned int i1row, unsigned int i2row )
{
	float res = 0;
	unsigned int ind1 = iax[i1row];
	unsigned int ind2 = iax[i2row];
	while (ind1 < iax[i1row+1] && ind2 < iax[i2row+1])
	{
		if (xi[ind1] == xi[ind2])
		{
			res += xv[ind1]*xv[ind2];
			++ind1;
			++ind2;
		}
		else if (xi[ind1] > xi[ind2])
		{
			++ind2;
		}
		else
		{
			++ind1;
		}
	}
	return res;
}
__device__ float get_norm(float *xv, int *xi, int *iax, float *yv, int *yi, int *iay, int i1row, int i2row )
{
	float res = 0;
	int ind1 = iax[i1row];
	int ind2 = iay[i2row];
	while (ind1 < iax[i1row+1] && ind2 < iay[i2row+1])
	{
		if (xi[ind1] == yi[ind2])
		{
			res += (xv[ind1]-yv[ind2])*(xv[ind1]-yv[ind2]);
			++ind1;
			++ind2;
		}
		else if (xi[ind1] > yi[ind2])
		{
			res += yv[ind2]*yv[ind2];
			++ind2;
		}
		else
		{
			res += xv[ind1]*xv[ind1];
			++ind1;
		}
	}
	while (ind1 < iax[i1row+1])
	{
		res += xv[ind1]*xv[ind1];
		++ind1;
	}
	while (ind2 < iay[i2row+1])
	{
		res += yv[ind2]*yv[ind2];
		++ind2;
	}
	return res;
}
/**
* Calculate kernel matrix row
* @param d_k device pointer to the kernel matrix
* @param tv device pointer to the train vectors
* @param gamma parameter gamma for RBF kernel
* @param nfeatures # of features
* @param irow  global index of kernel matrix row
* @param icache index of kernel matrix row in cache
* @param ntraining # of training samples
*/
__global__ void get_row(float *d_k, float *d_TVv, int *d_TVi, int *d_iaTV, unsigned int irow, unsigned int icache, int ntraining)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	while ( i < ntraining)
	{
		//d_k[icache*ntraining+i] = get_selfdot(d_TVv, d_TVi, d_iaTV, i) + get_selfdot(d_TVv, d_TVi, d_iaTV, irow) - 2*makedot(d_TVv, d_TVi, d_iaTV, i, irow);
		d_k[icache*ntraining+i] = get_norm(d_TVv, d_TVi, d_iaTV, d_TVv, d_TVi, d_iaTV, irow, i);
		i += gridsize;
	}
}
/**
* Make local reduce for classification task
* @param d_SV device pointer to the support vectors
* @param d_TV device pointer to the train vectors
* @param d_koef device pointer to array with y[i]*alfa[i] for each SV
* @param nSV # of support vectors
* @param ncol  # of features
* @param gamma parameter gamma for RBF kernel
* @param kernelcode kernel code
* @param result value of local reduction
*/
__global__ void reduction( float *d_SVv, int *d_SVi, int *d_iaSV, float *d_TVv, int *d_TVi, int *d_iaTV, float *d_koef, int nSV, int irowTV, int shift, float gamma, int kernelcode, float *result)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int blockSize = blockDim.x;
	unsigned int tid = threadIdx.x;
	extern __shared__  float reduct[];
	int TVi = d_iaTV[irowTV]-shift;
	reduct[tid]= 0;
	while (i < nSV)
	{
		if(kernelcode == 0)	
		{
			float val = 0;
			int SVi = d_iaSV[i];
			while (SVi < (d_iaSV[i+1]) && TVi < (d_iaTV[irowTV+1]-shift))
			{
				if (d_SVi[SVi] == d_TVi[TVi])
				{
					val += (d_TVv[TVi]-d_SVv[SVi])*(d_TVv[TVi]-d_SVv[SVi]);
					++SVi;
					++TVi;
				}
				else if (d_SVi[SVi] > d_TVi[TVi])
				{
					val += d_TVv[TVi]*d_TVv[TVi];
					++TVi;
				}
				else
				{
					val += d_SVv[SVi]*d_SVv[SVi];
					++SVi;
				}
			}
			while(SVi < (d_iaSV[i+1]))
			{
				val += d_SVv[SVi]*d_SVv[SVi];
				++SVi;
			}
			while(TVi < (d_iaTV[irowTV+1]-shift))
			{
				val += d_TVv[TVi]*d_TVv[TVi];
				++TVi;
			}
			reduct[tid] += d_koef[i]*expf(-gamma*val);
		}
		i += blockSize*gridDim.x;
	}
	__syncthreads();
		if(blockSize>=512)	{if(tid<256){reduct[tid] += reduct[tid + 256];}__syncthreads();}
	if(blockSize>=256)	{if(tid<128){reduct[tid] += reduct[tid + 128];}__syncthreads();}
	if(blockSize>=128)  {if(tid<64)	{reduct[tid] += reduct[tid + 64];}__syncthreads();}
	if(tid<32){	if(blockSize >= 64)	{reduct[tid] += reduct[tid + 32];}
				if(blockSize >= 32)	{reduct[tid] += reduct[tid + 16];}
				if(blockSize >= 16)	{reduct[tid] += reduct[tid + 8];}
				if(blockSize >= 8)	{reduct[tid] += reduct[tid + 4];}
				if(blockSize >= 4)	{reduct[tid] += reduct[tid + 2];}
				if(blockSize >= 2)	{reduct[tid] += reduct[tid + 1];}	}
	if(tid==0){	result[blockIdx.x]=reduct[0];}

}

__global__ void Update1(float *d_k, int *d_y, float *d_f, float *d_a, float *d_delta_a,
						unsigned int *d_I_global, unsigned int *d_I_cache, 
						float* d_param, int *d_active, int ntraining)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;//task number
	if (d_active[i] == 1)
	{
		int g_Iup = d_I_global[2*i];
		int g_Ilow = d_I_global[2*i+1];
		float alpha_up_old =d_a[i*ntraining+g_Iup];
		float alpha_low_old =d_a[i*ntraining+g_Ilow];
		float gamma;
		float eps = 0.000001;
		int s = d_y[g_Iup]*d_y[g_Ilow];
		float C = d_param[2*i];
		float L;
		float H;
		if (d_y[g_Iup] == d_y[g_Ilow])
			gamma = alpha_low_old + alpha_up_old;
		else
			gamma = alpha_low_old - alpha_up_old;


		if (s == 1)
		{
			L = max(0, gamma - C);
			H = min(C, gamma);
		}
		else
		{
			L = max(0, -gamma);
			H = min(C, C - gamma);
		}
		if (H <= L)
			d_active[i] = 0;

		float nu = 2*exp(-d_param[2*i+1]*d_k[d_I_cache[2*i+1]*ntraining+g_Iup]) - 2;
		float alpha_up_new;
		float alpha_low_new;
		if (nu < 0)
		{
			alpha_up_new = max(L, min(alpha_up_old - (d_y[g_Iup]*(d_f[i*ntraining+g_Ilow]-d_f[i*ntraining+g_Iup])/nu), H));
		}
		else
		{
			float slope= d_y[g_Iup]*(d_f[i*ntraining+g_Ilow]-d_f[i*ntraining+g_Iup]);
			float change= slope * (H-L);
			if(fabs(change)>0.0f)
			{
				if(slope>0.0f)
					alpha_up_new= H;
				else
					alpha_up_new= L;
			}
			else
				alpha_up_new= alpha_up_old;

			if( alpha_up_new > C - eps * C)
				alpha_up_new=C;
			else if (alpha_up_new < eps * C)
				alpha_up_new=0.0f;
		}
		if( fabs( alpha_up_new - alpha_up_old) < eps * ( alpha_up_new + alpha_up_old + eps))
			d_active[i] = 0;
		if (s == 1)
			alpha_low_new = gamma - alpha_up_new;
		else
			alpha_low_new = gamma + alpha_up_new;

		if( alpha_low_new > C - eps * C)
			alpha_low_new=C;
		else if (alpha_low_new < eps * C)
			alpha_low_new=0.0f;

		d_delta_a[2*i] = alpha_up_new - alpha_up_old;
		d_delta_a[2*i+1] = alpha_low_new - alpha_low_old;
		d_a[i*ntraining+g_Iup] = alpha_up_new;
		d_a[i*ntraining+g_Ilow] = alpha_low_new;

	}
}

__global__ void getdot(float *d_TVv, int *d_TVi, int *d_iaTV, int ntraining, float *dot)
{
	unsigned int gridsize = blockDim.x*gridDim.x;
	unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

	while ( i < ntraining)
	{
		dot[i] = get_selfdot(d_TVv, d_TVi, d_iaTV, i);
		i += gridsize;
	}
}