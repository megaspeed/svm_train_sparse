#ifndef _SVM_DATA_H_
#define _SVM_DATA_H_
#define MAXTHREADS 128
#define MAXBLOCKS 49152/MAXTHREADS
#define MAXBLOCKS_TV 49152/MAXTHREADS/MAXTHREADS
#define KMEM 0.65
#define TAU 0.001
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))

struct svm_sample
{
	int nTV;				/*# of test vectors/samples */
	int *l_TV;				/*	TV's labels				*/
	float *TV;				/*	TVs in CSR format	*/
	int *ind;
	int	*ia;			/*	ia[i] # of elements in ith sample vector*/
};

struct svm_model
{
	int nr_class;			/*	number of classes		*/
	int nSV;				/*	# of SV					*/
	float *SV;				/*	SVs in CSR format		*/
	int *ind;
	int		*ia;			/*	ia[i] # of elements in ith SV*/
	int nfeatures;
	float *sv_coef;		/*	SV's labels				*/
	float b;			/*	classification parametr	*/	
	int *label_set;		/*  intput lables			*/
	int svm_type;
	int kernel_type;
	float coef_d;
	float coef_gamma;
	float coef_b;
	float C;
	float *params;		/*	params C_i, gamma_i for RBF*/
	float* mass_b;
	int	ntasks;
};

#endif

