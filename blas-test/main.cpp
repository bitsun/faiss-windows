#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "cublas_wrapper.h"
 extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

void sgemm (const char *transa, const char *transb, int *m, int *
            n, int *k, const float *alpha, const float *a,
            int *lda, const float *b, int *
            ldb, float *beta, float *c, int *ldc);
 }
 
 ///initialize the matrix in column major order
 void init(float* matrix, int row, int column,bool bColumnMajor=true){
	 if(bColumnMajor){
	   for (int j = 0; j < column; j++){
			for (int i = 0; i < row; i++){
				matrix[j*row + i] = ((float)rand())/RAND_MAX;
			}
		}
	 }
	 else {
		 for (int i = 0; i < row; i++){
			for (int j = 0; j < column; j++){
				matrix[i*column + j] = ((float)rand())/RAND_MAX;
			}
		}
	 }
 }
 
 ////print the matrix in row major order
 void print(const char * name, const float* matrix, int row, int column,bool bStorageColumnMajor=true)
 {
   printf("Matrix %s has %d rows and %d columns:\n", name, row, column);
   if(bStorageColumnMajor){
	   for (int i = 0; i < row; i++){
		 for (int j = 0; j < column; j++){
		   printf("%.3f ", matrix[j*row + i]);
		 }
		 printf("\n");
	   }
	   printf("\n");
   }
   else {
	   //row major format
	   for (int i = 0; i < row; i++){
		 for (int j = 0; j < column; j++){
		   printf("%.3f ", matrix[i*column + j]);
		 }
		 printf("\n");
	   }
	   printf("\n");
   }
 }
 
 int main(int argc, char * argv[]){
	int rowsA, colsB, common;
	int i,j,k;
 
	if (argc != 4){
		 printf("Using defaults\n");
		rowsA = 1024*1024; colsB = 1024; common = 512;
	}
	else{
		 rowsA = atoi(argv[1]); colsB = atoi(argv[2]);common = atoi(argv[3]);
	}
	bool bColumnStorageFormat = false;

	float* A=new float[rowsA * common]; float* B=new float[common * colsB];
	float* C=new float[rowsA * colsB]; float* D=new float[rowsA * colsB];
	
	char transA = 'N', transB = 'N';
	float one = 1.0f, zero = 0.0f;
 
	srand(1);
 
	init(A, rowsA, common,bColumnStorageFormat); init(B, common, colsB,bColumnStorageFormat);
	std::cout<<"start "<<std::endl;
	Timer t;
	t.start();
	if(bColumnStorageFormat){
		sgemm(&transA, &transB, &rowsA, &colsB, &common, &one, A, 
			   &rowsA, B, &common, &zero, C, &rowsA);
	}
	else {
		int m=colsB, n=rowsA, k=common;
		int ldb = colsB,lda=common,ldc=colsB;
		sgemm(&transA, &transB, &m, &n, &k, &one, B, 
			   &ldb, A, &lda, &zero, C, &ldc);
	}
	t.stop();
	std::cout<<"mlk:"<<t.elapsedMilliseconds()<<std::endl;
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat=cublasCreate(&handle);
	t.start();
	CuBLASWrapper::MatMul(handle,A,B,D,rowsA,common,colsB,bColumnStorageFormat);
	t.stop();
	std::cout<<"cublas total:"<<t.elapsedMilliseconds()<<std::endl;
	//float* GT=new float[rowsA * colsB];
	//plain matrix computation
	if(bColumnStorageFormat){
		for(i=0;i<colsB;i++){
			for(j=0;j<rowsA;j++){
				//D[i*rowsA+j]=0;
				float sum=0;
				for(k=0;k<common;k++){
					sum+=A[k*rowsA+j]*B[k+common*i];
				}
				//GT[i*rowsA+j] = sum;
				if (abs(C[i*rowsA + j] - sum)/abs(sum) >= 1e-4) {
					std::cout<<"error -> row: "<<j<<" col:"<<i<<" GT:"<<sum<<" mkl blas "<<C[i*rowsA + j]<<std::endl;
				}
				if (abs(D[i*rowsA + j] - sum)/abs(sum) >= 1e-4) {
					std::cout<<"error -> row: "<<j<<" col:"<<i<<" GT:"<<sum<<" cuda blas "<<D[i*rowsA + j]<<std::endl;
				}
			}
		}
	}
	else {
		for (i = 0; i < rowsA; ++i) {
			for (j = 0; j < colsB; ++j) {
				float v_ij=0;
				for(k=0;k<common;k++){
					v_ij+=A[i*common+k]*B[j+colsB*k];
				}
				//GT[i*colsB+j]=v_ij;
				if (abs(C[i*colsB+j] - v_ij)/abs(v_ij) >= 1e-4) {
					std::cout<<"error -> row: "<<i<<" col:"<<j<<" GT:"<<v_ij<<" mkl blas "<<C[i*colsB+j]<<std::endl;
				}
				if (abs(D[i*colsB+j] - v_ij)/abs(v_ij) >= 1e-4) {
					std::cout<<"error -> row: "<<i<<" col:"<<j<<" GT:"<<v_ij<<" cuda blas "<<D[i*colsB+j]<<std::endl;
				}
			}
		}
	}
	printf("done\n");
   //print("A", A, rowsA, common,bColumnStorageFormat); print("B", B, common, colsB,bColumnStorageFormat);
   //print("C", C, rowsA, colsB,bColumnStorageFormat); print("D", D, rowsA, colsB,bColumnStorageFormat);
 
   return 0;
 }