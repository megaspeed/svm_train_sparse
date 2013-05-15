#include "common.cpp"
#include <cuda_runtime_api.h>
#include <float.h>
#include "device_launch_parameters.h"
#include "kernels.cu"
# define cudaCheck\
 {\
 cudaError_t err = cudaGetLastError ();\
 if ( err != cudaSuccess ){\
 printf(" cudaError = '%s' \n in '%s' %d\n", cudaGetErrorString( err ), __FILE__ , __LINE__ );\
 exit(0);}}


void classificator(svm_model *model, svm_sample *test, float *rate)
{
	int nTV = test->nTV;		//# of test samples
	int nSV = model->nSV;		// # of support vectors
	int nSVelem = model->ia[nSV];
	int nTVelem = test->ia[nTV];
	float *d_SVv = 0;
	cudaMalloc((void**)&d_SVv, nSVelem*sizeof(float)); cudaCheck
	int *d_SVi = 0;
	cudaMalloc((void**)&d_SVi, nSVelem*sizeof(int)); cudaCheck
	cudaMemcpy(d_SVv, model->SV, nSVelem*sizeof(float), cudaMemcpyHostToDevice); cudaCheck
	cudaMemcpy(d_SVi, model->ind, nSVelem*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	float *d_TVv = 0;
	cudaMalloc((void**)&d_TVv, nTVelem*sizeof(float)); cudaCheck
	int *d_TVi = 0;
	cudaMalloc((void**)&d_TVi, nTVelem*sizeof(int)); cudaCheck
	cudaMemcpy(d_TVv, test->TV, nTVelem*sizeof(float), cudaMemcpyHostToDevice); cudaCheck
	cudaMemcpy(d_TVi, test->ind, nTVelem*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	int *d_iaSV = 0;
	cudaMalloc((void**)&d_iaSV, (nSV+1)*sizeof(int)); cudaCheck
	cudaMemcpy(d_iaSV, model->ia, (nSV+1)*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	int *d_iaTV = 0;
	cudaMalloc((void**)&d_iaTV, (nTV+1)*sizeof(int)); cudaCheck
	cudaMemcpy(d_iaTV, test->ia, (nTV+1)*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	float *d_l_SV = 0;
	cudaMalloc((void**) &d_l_SV, nSV*sizeof(float));cudaCheck
	cudaMemcpy(d_l_SV, model->sv_coef, nSV*sizeof(float),cudaMemcpyHostToDevice);cudaCheck

	size_t remainingMemory;
	size_t totalMemory;
	cudaMemGetInfo(&remainingMemory, &totalMemory); cudaCheck
	printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);



	int nthreads = MAXTHREADS;
	int nblocksSV = min(MAXBLOCKS, (nSV + nthreads - 1)/nthreads);
	
	// Allocate device memory for F
	float* h_fdata= (float*) malloc(nblocksSV*sizeof(float));
	float* d_fdata=0;
	cudaMalloc((void**) &d_fdata, nblocksSV*sizeof(float));cudaCheck
	int available_memory = (int)(remainingMemory*KMEM/sizeof(float));
	if (nTVelem <= available_memory/2){	available_memory = 2*nTVelem; }
	//cudaMalloc((void**)&d_TVv, available_memory*sizeof(float)); cudaCheck
	//cudaMalloc((void**)&d_TVi, available_memory*sizeof(int)); cudaCheck
	int offset = 0;
	int *cache_size = NULL; // # of TVs in cache
	int num_of_parts;
	get_cached_rows(test->ia, available_memory, nTV, &cache_size, &num_of_parts);

	int* h_l_estimated = (int*)malloc(nTV*sizeof(int));

	for (int parti = 0; parti < num_of_parts; parti++)
	{
		//int nelem = test->ia[offset + cache_size[parti]]-test->ia[offset];
		//cudaMemcpy(d_TVv, &test->TV[test->ia[offset]], nelem*sizeof(float), cudaMemcpyHostToDevice); cudaCheck
		//cudaMemcpy(d_TVi, &test->ind[test->ia[offset]], nelem*sizeof(int), cudaMemcpyHostToDevice); cudaCheck
		for (int i = 0; i < cache_size[parti]; i++)
		{				
			reduction<<<nblocksSV, MAXTHREADS, MAXTHREADS*sizeof(float)>>>(d_SVv, d_SVi, d_iaSV, d_TVv, d_TVi, d_iaTV, d_l_SV, nSV, i+offset, test->ia[offset], model->coef_gamma, 0, d_fdata);cudaCheck
				
			cudaMemcpy(h_fdata, d_fdata, nblocksSV*sizeof(float), cudaMemcpyDeviceToHost); cudaCheck

			float sum = 0;
			for (int k = 0; k < nblocksSV; k++)
				sum += h_fdata[k];

			sum -= model->b;
			if (sum > 0)
			{
				h_l_estimated[i + offset] = 1;
			}
			else
			{
				h_l_estimated[i + offset] = -1;
			}
		}
		offset += cache_size[parti];
	}
	
	cudaFree(d_SVi);cudaCheck
	cudaFree(d_SVv);cudaCheck
	cudaFree(d_TVi);cudaCheck
	cudaFree(d_TVv);cudaCheck
	cudaFree(d_iaSV);cudaCheck
	cudaFree(d_iaTV);cudaCheck
	cudaFree(d_l_SV);cudaCheck
	cudaFree(d_fdata);cudaCheck
	cudaDeviceReset();cudaCheck

	int errors=0;
	for (int i=0; i<nTV; i++)
	{
		if( test->l_TV[i]!=h_l_estimated[i])
		{
			errors++;
		}
	}
	*rate = (float)(nTV - errors)/nTV;
	
	free(h_l_estimated);
	free(h_fdata);
	printf("# of testing samples %i, # errors %i, Rate %f\n", nTV, errors, 100* (float)(nTV-errors)/nTV);
}

void Reduce_step(int *d_y, float *d_a, float *d_f, float *d_B, unsigned int *d_I, float *param, int ntraining, int nblocks,
				 float *h_B, unsigned int *h_I, float* h_B_global, unsigned int *h_I_global, int *active, int ntasks)
{
	int smem = MAXTHREADS*(sizeof(float) + sizeof(int));
	for (int itask = 0; itask < ntasks; itask++)
	{
		if (active[itask] == 1)
		{
			Local_Reduce_Min<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks], &d_I[itask*2*nblocks], &param[2*itask], ntraining);

			Local_Reduce_Max<<<nblocks, MAXTHREADS, smem>>>(d_y, &d_a[itask*ntraining], &d_f[itask*ntraining], &d_B[itask*2*nblocks+nblocks], &d_I[itask*2*nblocks+nblocks], &param[2*itask], ntraining);
					//cudaDeviceSynchronize();
		}
	}
	cudaMemcpy(h_B, d_B, ntasks*2*nblocks*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_I, d_I, ntasks*2*nblocks*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	for (int itask = 0; itask < ntasks; itask++)
	{
		if (active[itask] == 1)
		{
			// Global reduction
			float global_Bup = h_B[itask*2*nblocks];
			float global_Blow = h_B[itask*2*nblocks+nblocks];
			int global_Iup = h_I[itask*2*nblocks];
			int global_Ilow = h_I[itask*2*nblocks+nblocks];

			for (int i = 1; i < nblocks; i++)
			{
				if (h_B[itask*2*nblocks+i] < global_Bup)
				{
					global_Bup = h_B[itask*2*nblocks+i];
					global_Iup = h_I[itask*2*nblocks+i];
				}
				if (h_B[itask*2*nblocks+nblocks + i] > global_Blow)
				{
					global_Blow = h_B[itask*2*nblocks+nblocks + i];
					global_Ilow = h_I[itask*2*nblocks+nblocks + i];
				}
			}

			h_B_global[itask*2] = global_Bup;
			h_B_global[itask*2+1] = global_Blow;
			h_I_global[itask*2] = global_Iup;
			h_I_global[itask*2+1] = global_Ilow;
		}
	}
}

void cross_validation(svm_sample *train, svm_model *model)
{
	int ntasks = model->ntasks;
	int nTV = train->nTV;
	int nTVelem = train->ia[nTV];
	//Grid configuration
	int nthreads = MAXTHREADS;
	int nblocks = min(MAXBLOCKS, (nthreads + nTV - 1)/nthreads);
	int task_threads = min(1024/MAXTHREADS, ntasks);
	int task_blocks = (ntasks + task_threads - 1)/task_threads;

	float *d_TVv = 0;
	cudaMalloc((void**)&d_TVv, nTVelem*sizeof(float)); cudaCheck
	int *d_TVi = 0;
	cudaMalloc((void**)&d_TVi, nTVelem*sizeof(int)); cudaCheck
	cudaMemcpy(d_TVv, train->TV, nTVelem*sizeof(float), cudaMemcpyHostToDevice); cudaCheck
	cudaMemcpy(d_TVi, train->ind, nTVelem*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	int *d_iaTV = 0;
	cudaMalloc((void**)&d_iaTV, (nTV+1)*sizeof(int)); cudaCheck
	cudaMemcpy(d_iaTV, train->ia, (nTV+1)*sizeof(int), cudaMemcpyHostToDevice); cudaCheck

	float *d_params = 0;
	cudaMalloc((void**) &d_params, 2*ntasks*sizeof(float));
	cudaMemcpy(d_params, model->params, 2*ntasks*sizeof(float),cudaMemcpyHostToDevice);

	int *d_y = 0;// binary labels
	cudaMalloc((void**) &d_y, nTV*sizeof(int));
	cudaMemcpy(d_y, train->l_TV, nTV*sizeof(int),cudaMemcpyHostToDevice);

	float *d_a = 0; //alphas
	cudaMalloc((void**) &d_a, ntasks*nTV*sizeof(float));

	float *d_f = 0;//object functions
	cudaMalloc((void**) &d_f, ntasks*nTV*sizeof(float));

	float *d_dot = 0;//self dot TV
	cudaMalloc((void**) &d_dot, nTV*sizeof(float));
	getdot<<<nblocks, nthreads>>>(d_TVv, d_TVi, d_iaTV, nTV, d_dot);

	//locally reduced thresholds {Bup:Blow}
	float *h_B = (float*)malloc(2*nblocks*ntasks*sizeof(float));
	float *d_B = 0;
	cudaMalloc((void**) &d_B, 2*nblocks*ntasks*sizeof(float));

	//indeces of locally reduced Lagrange multipliers {Iup:Ilow}
	unsigned int *h_I = (unsigned int*)malloc(2*nblocks*ntasks*sizeof(unsigned int));
	unsigned int *d_I = 0; 
	cudaMalloc((void**) &d_I, 2*nblocks*ntasks*sizeof(unsigned int));

	//global tresholds {Bup:Blow}
	float *h_B_global = (float*)malloc(2*ntasks*sizeof(float));

	unsigned int *h_I_global = (unsigned int*)malloc(2*ntasks*sizeof(unsigned int));
	unsigned int *d_I_global = 0; 
	cudaMalloc((void**) &d_I_global, 2*ntasks*sizeof(unsigned int));
	unsigned int *h_I_cache = (unsigned int*)malloc(2*ntasks*sizeof(unsigned int));
	unsigned int *d_I_cache = 0; 
	cudaMalloc((void**) &d_I_cache, 2*ntasks*sizeof(unsigned int));

	float *d_delta_a = 0;
	cudaMalloc((void**) &d_delta_a, 2*ntasks*sizeof(float));

	int *h_active = (int*)malloc(ntasks*sizeof(int));
	for (int i = 0; i < ntasks; i++)
		h_active[i] = 1;

	int *d_active = 0;
	cudaMalloc((void**) &d_active, ntasks*sizeof(int));
	cudaMemcpy(d_active, h_active, ntasks*sizeof(int),cudaMemcpyHostToDevice);


	initialization<<<dim3(nblocks, task_blocks), dim3(nthreads, task_threads)>>>(d_a, d_f, d_y, nTV, ntasks);
	cudaDeviceSynchronize();
	Reduce_step(d_y, d_a, d_f, d_B, d_I, d_params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);

	unsigned int remainingMemory;
	unsigned int totalMemory;
	cudaMemGetInfo(&remainingMemory, &totalMemory);
	printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);
	int sizeOfCache = remainingMemory/(nTV*sizeof(float));
	sizeOfCache = (int)((float)sizeOfCache*KMEM);
	if (nTV < sizeOfCache)
		sizeOfCache = nTV;
	printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, nTV*sizeof(float));
	//sizeOfCache = 9000;
	float *d_k = 0;// gramm matrix
	cudaMalloc((void**) &d_k, sizeOfCache*nTV*sizeof(float));

	int iter = 0;
	std::list<std::pair<unsigned int,unsigned int> > cache;
	
	while (chech_condition(h_B_global, h_active, ntasks, iter))
	{
		++iter;	
		for (int itask = 0; itask < ntasks; itask++)
		{
			if (h_active[itask] == 1)
			{
				if(check_cache(h_I_global[2*itask], &h_I_cache[2*itask], &cache, sizeOfCache))		//Iup - second
				{
					get_row<<<nblocks, nthreads>>>(d_k, d_TVv, d_TVi, d_iaTV, d_dot, h_I_global[2*itask], h_I_cache[2*itask], nTV);
				}
				if(check_cache(h_I_global[2*itask+1], &h_I_cache[2*itask+1], &cache, sizeOfCache))//Ilow - fist
				{
					get_row<<<nblocks, nthreads>>>(d_k, d_TVv, d_TVi, d_iaTV, d_dot, h_I_global[2*itask+1], h_I_cache[2*itask+1], nTV);
				}
			}
		}
		cudaMemcpy(d_active, h_active, ntasks*sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_cache, h_I_cache, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);
		cudaMemcpy(d_I_global, h_I_global, ntasks*2*sizeof(unsigned int),cudaMemcpyHostToDevice);

		Update<<<1,ntasks>>>(d_k, d_y, d_f, d_a, d_delta_a, d_I_global, d_I_cache, d_params, d_active, nTV);
		cudaDeviceSynchronize();
		Map<<<dim3(nblocks, task_blocks), dim3(nthreads, task_threads)>>>(d_f, d_k, d_y, d_delta_a, d_I_global, d_I_cache, d_params, d_active, nTV);
		cudaDeviceSynchronize();
		Reduce_step(d_y, d_a, d_f, d_B, d_I, d_params, nTV, nblocks, h_B, h_I, h_B_global, h_I_global, h_active, ntasks);
	}
	printf("All tasks convergented in %f\n", cuGetTimer());

	model->sv_coef = (float*)malloc(nTV*ntasks*sizeof(float));
	cudaMemcpy(model->sv_coef, d_a, nTV*ntasks*sizeof(float), cudaMemcpyDeviceToHost);

	model->mass_b = (float*)malloc(ntasks*sizeof(float));
	for (int itask = 0; itask < model->ntasks; itask++)
	{
		model->mass_b[itask] = (h_B_global[2*itask+1]+h_B_global[2*itask])/2;
	}
	cudaDeviceReset();
}

void classification( svm_sample *train, svm_sample *test, svm_model *model, FILE *output)
{
	int ntraining = train->nTV;
	int nTVelem = train->ia[ntraining];
	float *buf_l = model->sv_coef;
	float rate;
	float max_rate = 0;
	int max_rate_ind;

	for (int itask = 0; itask < model->ntasks; itask++)
	{
		float *SVv = (float*)malloc(nTVelem*sizeof(float));
		int *SVi = (int*)malloc(nTVelem*sizeof(int));
		int *iaSV = (int*)malloc((ntraining+1)*sizeof(int));
		float *sv_coef = (float*)malloc(ntraining*sizeof(float));
		model->sv_coef = &buf_l[itask*ntraining];
		int nSV = 0;
		iaSV[0] = 0;
		int k;
		for (int i = 0; i < ntraining; i++)
		{
			if (model->sv_coef[i] != 0)
			{
				k = 0;
				sv_coef[nSV] = train->l_TV[i]*model->sv_coef[i];
				for (int j = train->ia[i]; j < train->ia[i+1]; j++, k++)
				{
					SVv[iaSV[nSV]+k] = train->TV[j];
					SVi[iaSV[nSV]+k] = train->ind[j];
				}
				iaSV[nSV+1] = iaSV[nSV] + k;
				++nSV;
			}
		}
		if (nSV == 0)
		{
			printf("Task %d has bad parameters C=%f and gamma=%f #SV=%d\n",itask, model->params[2*itask], model->params[2*itask+1], nSV);
			continue;
		}
		model->nSV = nSV;
		model->b = model->mass_b[itask];
		model->C = model->params[2*itask];
		model->coef_gamma = model->params[2*itask+1];
		model->sv_coef=(float*)realloc(sv_coef, nSV*sizeof(float));
		model->SV=(float*)realloc(SVv, iaSV[nSV]*sizeof(float));
		model->ind=(int*)realloc(SVi, iaSV[nSV]*sizeof(int));
		model->ia=(int*)realloc(iaSV, (nSV+1)*sizeof(int));
		classificator(model, test, &rate);
		save_model(output, model);
		if (max_rate < rate)
		{
			max_rate = rate;
			max_rate_ind = itask;
		}
		free(model->sv_coef);
		free(model->SV);
		printf("Task %d occuracy is %f with C=%f and gamma=%f #SV=%d\n",itask, rate, model->params[2*itask], model->params[2*itask+1], nSV);
	}
	printf("best occuracy is %f with C=%f and gamma=%f\n", max_rate, model->params[2*max_rate_ind], model->params[2*max_rate_ind+1]);
}

int main(int argc, char **argv)
{
	FILE *input_train, *input_test, *output;
	if (argc==1)
	{
		argc = 5;
		argv[1] = "C:\\Data\\b.txt";
		argv[2] = "C:\\Data\\b.model";
		argv[3] = "C:\\Data\\b.txt";
		argv[4] = "10";
		argv[1] = "C:\\Data\\a9a";
		argv[2] = "C:\\Data\\a9a.model";
		argv[3] = "C:\\Data\\a9a.t";
		argv[4] = "123";
		argv[1] = "C:\\Data\\cov";
		argv[2] = "C:\\Data\\cov.model";
		argv[3] = "C:\\Data\\cov.t";
		argv[4] = "54";
		//argv[1] = "C:\\Data\\mushrooms";
		//argv[2] = "C:\\Data\\mushrooms.model";
		//argv[3] = "112";
		//argv[1] = "C:\\Data\\ijcnn1";
		//argv[2] = "C:\\Data\\ijcnn1.model";
		//argv[3] = "22";

	}
	if(argc<5)
		exit_with_help();
	struct svm_model *model = (svm_model*)malloc(sizeof(svm_model));
	struct svm_sample *train = (svm_sample*)malloc(sizeof(svm_sample));
	struct svm_sample *test = (svm_sample*)malloc(sizeof(svm_sample));
	sscanf(argv[4],"%d",&model->nfeatures);

	if((input_train = fopen(argv[1],"r")) == NULL)
	{
		fprintf(stderr,"can't open training file %s\n",argv[1]);
		exit(1);
	}

	if((output = fopen(argv[2],"w")) == NULL)
	{
		fprintf(stderr,"can't create model file %s\n",argv[2]);
		exit(1);
	}
	if((input_test = fopen(argv[3],"r")) == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",argv[1]);
		exit(1);
	}
	float percent = 0.8;
	float rate;
	set_model_param(model, 1, 1, 0.04, 1);
	converg_time= (float*)malloc(model->ntasks*sizeof(float));
	for (int itask = 0; itask < model->ntasks; itask++)
		converg_time[itask] = 0;
	parse_TV(input_test, test);
	parse_TV(input_train, train);
	set_labels(test, model);
	set_labels(train, model);
	//balabce_data(train, test, percent);
	cuResetTimer();
	cross_validation(train, model);
	classification(train, test, model, output);
	printf("Cache: hits %d miss %d percent of hits %f\n", cache_hit, cache_miss, (float)cache_hit/cache_miss*100);
	printf("Total time %f\n", cuGetTimer());
	cudaDeviceReset();
	return 0;
}
