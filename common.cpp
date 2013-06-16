#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "svm_data.h"
#include <list>
#include <algorithm>
int cache_hit = 0;
int cache_miss = 0;
float *converg_time;
#ifdef _WIN32

#include <windows.h>

static LARGE_INTEGER t;
static float         f;
static int           freq_init = 0;

void cuResetTimer(void) {
  if (!freq_init) {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    f = (float) freq.QuadPart;
    freq_init = 1;
  }
  QueryPerformanceCounter(&t);
}

float cuGetTimer(void) {
  LARGE_INTEGER s;
  float d;
  QueryPerformanceCounter(&s);

  d = ((float)(s.QuadPart - t.QuadPart)) / f;

  return (d*1000.0f);
}

#else

#include <sys/time.h>

static struct timeval t;

/**
 * Resets timer
 */
void cuResetTimer() {
  gettimeofday(&t, NULL);
}


/**
 * Gets time since reset
 */
float cuGetTimer() { // result in miliSec
  static struct timeval s;
  gettimeofday(&s, NULL);

  return (s.tv_sec - t.tv_sec) * 1000.0f + (s.tv_usec - t.tv_usec) / 1000.0f;
}

#endif


/**
* Set labels to {1;-1}
*/
void set_labels(svm_sample *train, svm_model *model)
{
	model->nr_class = 2;
	model->label_set = (int*)malloc(2*sizeof(int));
	model->SVperclass = (int*)malloc(2*sizeof(int));
	model->SVperclass[0] = 0;
	model->label_set[0] = 1;
	model->label_set[1] = -1;
	int buf = train->l_TV[0];
	for (int i = 1; i < train->nTV; i++)
	{
		if (buf < train->l_TV[i])
		{
			model->label_set[0] = train->l_TV[i];
			model->label_set[1] = buf;
			break;
		}
		if (buf > train->l_TV[i])
		{
			model->label_set[0] = buf;
			model->label_set[1] = train->l_TV[i];
			break;
		}
		++i;
	}

	for (int i = 0; i < train->nTV; i++)
	{
		if (train->l_TV[i] == model->label_set[0])
			{
				train->l_TV[i] = 1;
				++model->SVperclass[0];
			}
		else
			train->l_TV[i] = -1;
	}
	model->SVperclass[1] = train->nTV - model->SVperclass[0];
}

static char* readline(FILE *input, char** line, unsigned int *max_line_len)
{
	int len;
	if(fgets(*line,*max_line_len,input) == NULL)
		return NULL;

	while(strrchr(*line,'\n') == NULL)
	{
		*max_line_len *= 2;
		*line = (char *) realloc(*line,*max_line_len);
		len = (int) strlen(*line);
		if(fgets(*line+len,*max_line_len-len,input) == NULL)
			break;
	}
	return *line;
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}
/**
* Parses sample vectors from file in the libsvm format (only 2 classes)
*/
int parse_TV(FILE* fp, svm_sample *train)
{
	int elements = 0;		// # of presented features
	int l = 0;
	unsigned int max_line_len = 1024;
	char *line = (char*)malloc(max_line_len*sizeof(char));
	char *p,*endptr,*idx,*val;

	while(readline(fp, &line, &max_line_len)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
		++l;
	}
	train->nTV = l;
	rewind(fp);

	train->l_TV = (int*)malloc(l*sizeof(int));
	train->TV = (float*)malloc(elements*sizeof(float));
	train->ind = (int*)malloc(elements*sizeof(int));
	train->ia = (int*)malloc((l+1)*sizeof(int));
	int *index = train->ind;
	float* value = train->TV;
	int *ia = train->ia;
	ia[0] = 0;

	int j = 0;
	for(int i = 0; i < l; i++)
	{
		readline(fp, &line, &max_line_len);

		p = strtok(line, " \t");
		train->l_TV[i] = (int)strtod(p,&endptr);

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			index[j] = (int) strtol(idx,&endptr,10)-1;
			value[j] = (float)strtod(val,&endptr);
			++j;
		}
		ia[i+1] = j;
	}
	free(line);
	if (fclose(fp) != 0)
		return 1;

	return 0;
}
int read_model(const char* model_file_name, svm_model *model)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return 0;
	const char *svm_type_table[] = { "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",0 };
	const char *kernel_type_table[] = { "rbf","linear","polynomial","sigmoid","precomputed",0 };
	// read parameters
	model->SV = NULL;
	model->ind = NULL;
	model->ia = NULL;
	model->sv_coef = NULL;
	model->label_set = NULL;
	model->b = NULL;
	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0; svm_type_table[i]!=0;i++)
			{
				if(strcmp(svm_type_table[i],cmd)==0)
				{
					model->svm_type=i;
					break;
				}
			}
			if(svm_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown svm type.\n");
				return 0;
			}
		}
		else if(strcmp(cmd,"kernel_type")==0)
		{		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++)
			{
				if(strcmp(kernel_type_table[i],cmd)==0)
				{
					model->kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown kernel function.\n");
				return 0;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%f",&model->coef_d);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%f",&model->coef_gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%f",&model->coef_b);
		else if(strcmp(cmd,"nr_class")==0)
			fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->nSV);
		else if(strcmp(cmd,"rho")==0)
			fscanf(fp,"%f",&model->b);
		else if(strcmp(cmd,"label")==0)
		{
			int n = model->nr_class;
			model->label_set = (int*)malloc(n*sizeof(int));
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label_set[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0)
		{
			int n = model->nr_class;
			model->SVperclass = (int*)malloc(n*sizeof(int));
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->SVperclass[i]);
		}
		else if(strcmp(cmd,"SV")==0)
		{
			while(1)
			{
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			//free_model(model);
			return 0;
		}
	}
	// read sv_coef and SV

	int elements = 0;		// # of presented features
	long pos = ftell(fp);

	unsigned int max_line_len = 1024;
	char *line = (char*)malloc(max_line_len*sizeof(char));
	char *p,*endptr,*idx,*val;

	while(readline(fp, &line, &max_line_len)!=NULL)
	{
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
	int l = model->nSV;
	model->sv_coef = (float*)malloc(l*sizeof(float));
	int i;
	model->SV = (float*)malloc(elements*sizeof(float));
	model->ind = (int*)malloc(elements*sizeof(int));
	model->ia = (int*)malloc((model->nSV+1)*sizeof(int));

	int *index = model->ind;
	float* value = model->SV;
	int *ia = model->ia;
	int j = 0;
	ia[0] = 0;
	for(i=0;i<l;i++)
	{
		readline(fp, &line, &max_line_len);
		p = strtok(line, " \t");
		model->sv_coef[i] = (float)strtod(p,&endptr);

		while(1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			index[j] = (int) strtol(idx,&endptr,10)-1;
			value[j] = (float)strtod(val,&endptr);
			++j;
		}		
		ia[i+1] = j;
	}
	free(line);
	if (ferror(fp) != 0 || fclose(fp) != 0)
	{
		return 1;
	}
	return 0;
}
void exit_with_help()
{
	printf("Usage: svm-predict test_file model_file #_of_features\n");
	exit(1);
}
int save_model(FILE *fp, const svm_model *model)
{
	const char *svm_type_table[] = { "c_svc","nu_svc","one_class","epsilon_svr","nu_svr",0 };
	const char *kernel_type_table[] = { "rbf","linear","polynomial","sigmoid","precomputed",0 };
	fprintf(fp,"svm_type %s\n", svm_type_table[model->svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[model->kernel_type]);

	if(model->kernel_type == 2)
		fprintf(fp,"degree %d\n", model->coef_d);

	if(model->kernel_type == 2 || model->kernel_type == 0 || model->kernel_type == 3)
		fprintf(fp,"gamma %g\n", model->coef_gamma);

	if(model->kernel_type == 2 || model->kernel_type == 3)
		fprintf(fp,"coef0 %g\n", model->coef_b);

	int nr_class = model->nr_class;
	int l = model->nSV;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	fprintf(fp,"rho %f\n",model->b);
	
	if(model->label_set)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label_set[i]);
		fprintf(fp, "\n");
	}

	if(model->SVperclass)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->SVperclass[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	float *sv_coef = model->sv_coef;
	float *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j*nr_class+i]);

		for (int j = model->ia[i]; j < model->ia[i+1]; j++)
		{
			fprintf(fp,"%d:%.8g ", model->ind[j]+1, SV[j]);
		}
		fprintf(fp, "\n");
	}

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}
/**
* Manage cache 
*/
bool check_cache(unsigned int irow, unsigned int *cached_row, std::list<std::pair<unsigned int,unsigned int>> *cache, int cache_size)
{
	unsigned int pos = 0;
	std::list<std::pair<unsigned int, unsigned int>>::iterator findIter;
	for (findIter = cache->begin(); findIter != cache->end(); ++findIter, ++pos)
	{
		if (irow == findIter->first)
		{
			*cached_row = findIter->second;
			cache->remove(*findIter);
			cache->push_front(std::make_pair(irow, *cached_row));
			++cache_hit;
			return false;
		}
	}

	if (cache->size() == cache_size)
	{
		*cached_row = (--findIter)->second;
		cache->pop_back();
	}
	else
	{
		*cached_row = pos;
	}
	++cache_miss;
	cache->push_front(std::make_pair(irow, *cached_row));
	return true;	
}
/**
* Return false if all tasks have converged
*/
bool chech_condition(float* B, int *active_task, int ntasks, int iter)
{
	bool run = false;
	for (int i = 0; i < ntasks; i++)
	{
		if (B[2*i+1] <= B[2*i] + 2*TAU)
		{
			active_task[i] = 0;
			//if(!converg_time[i]){
			//	converg_time[i]=cuGetTimer();
			//	printf("Task %d has convergent in %f on iter=%d\n", i, converg_time[i], iter);
			//}
		}
		run = run||active_task[i];
	}
	return run;
}
/**
* Generate parameters set
* @param model host pointer to model struct
* @param cbegin first value of parameter C
* @param c_col total # of C(i) where C(i)=C(i-1)/2
* @param gbegin first value of parameter gamma in RBF kernel
* @param g_col total # of gamma(i) where gamma(i)=gamma(i-1)/2
* gamma[0] always equal 1/nfeatureas
*/
void set_model_param(svm_model *model, float cbegin, int c_col, float gbegin, int g_col)
{
	float *C = (float*)malloc((c_col)*sizeof(float));
	float *gamma = (float*)malloc((g_col)*sizeof(float));
	C[0] = cbegin;
	for (int i = 1; i < c_col; i++)
	{
		C[i] = C[i-1]/2;
	}

	gamma[0] = max(1./model->nfeatures, 0.0001);
	if(g_col > 1)
		gamma[1] = gbegin;
	for (int i = 2; i < g_col; i++)
	{
		gamma[i] = gamma[i-1]/2;
	}

	model->ntasks = c_col*g_col;
	model->params = (float*)malloc(model->ntasks*2*sizeof(float));
	for (int i = 0; i < c_col; i++)
	{
		for (int j = 0; j < g_col; j++)
		{
			model->params[2*i*g_col+2*j] = C[i];
			model->params[2*i*g_col+2*j+1] = gamma[j];
		}
	}
	model->kernel_type = 0;
	model->svm_type = 0;
	free(C);
	free(gamma);
}

void get_cached_rows(int *ai, int cache_size, int n, int **num_rows, int *num_parts)
{
	int buf = 0;
	int k = 0;
	int starti = 0;
	int *parts = (int*) malloc(n*sizeof(int));
	for (int i = 0; i < n ; i++)
	{ 
		if ((ai[i+1] - buf) > cache_size)
		{
			parts[k] = i - starti;
			++k;
			starti = i;
			buf = ai[i];
		}
	}
	if (ai[n] != buf)
	{
		parts[k] = n - starti;
		++k;
	}
	*num_parts = k;
	*num_rows = (int*)realloc(parts, k*sizeof(int));
}
//Swap i,j label
void swap_l(int *l, int i, int j)
{
	int buf = l[i];
	l[i] = l[j];
	l[j] = buf;
}

void set_folds(svm_sample *train, svm_sample *test, int nfolds, int nfeatures)
{
	svm_sample *folds = (svm_sample*)malloc(sizeof(svm_sample));
	int n = train->nTV;
	int m = n/nfolds;
	int total_elem = nfolds*m;
	folds->l_TV = (int*)malloc(total_elem*sizeof(int));
	folds->TV = (float*)malloc(total_elem*nfeatures*sizeof(float));
	int shift = m*nfeatures;
	for (int i = 0; i < m; i++)
	{
		for (int ifold = 0; ifold < nfolds; ifold++)
		{
			folds->l_TV[ifold*m+i] = train->l_TV[i*nfolds+ifold];
			for (int k = 0; k < nfeatures; k++)
			{
				folds->TV[ifold*shift+i*nfeatures+k] = train->TV[(i*nfolds+ifold)*nfeatures+k];
			}
		}
	}
	free(train->TV);
	free(train->l_TV);
	train->nTV = (nfolds-1)*m;
	test->nTV = m;
	train->TV = folds->TV;
	train->l_TV = folds->l_TV;
	test->TV = &folds->TV[train->nTV*nfeatures];
	test->l_TV = &folds->l_TV[train->nTV];
}
