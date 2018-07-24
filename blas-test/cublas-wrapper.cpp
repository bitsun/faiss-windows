#include "cublas_wrapper.h"
#include "cuda_runtime.h"
#include <exception>
#include <iostream>
bool CuBLASWrapper::MatMul(cublasHandle_t& handle,float*A, float* B, float* C, const int rowA, const int colArowB, const int colB,bool bColumnMajorFormat) {                                         
		cudaError_t cudastat;
		
		int sizeA=rowA*colArowB;                    
		int sizeB=colArowB*colB;
		int sizeC=rowA*colB;
		

		float* d_A;  
		float* d_B; 
		float* d_C;  //result
		
		cudastat=cudaMalloc((void**)&d_A,sizeA*sizeof(float)); 
		cudastat=cudaMalloc((void**)&d_B,sizeB*sizeof(float));
		cudastat=cudaMalloc((void**)&d_C,sizeC*sizeof(float));

		cudaMemcpy(d_A,A,sizeof(float)*sizeA,cudaMemcpyHostToDevice);  //copy A to device d_A
		cudaMemcpy(d_B,B,sizeof(float)*sizeB,cudaMemcpyHostToDevice);   //copy B to device d_B

		float alf=1.0; //check
		float beta=0;
		
		//in the original CuBLAS documentation
		// and A , B and C are matrices stored in column-major format with dimensions 
		// op ( A ) m × k , op ( B ) k × n and C m × n
		//in our case ,op is always none, in our case op(A) = B' -> colB x colArowB, op(B) = A' -> colArowB x rowA
		//int m=colB, n=rowA, k=colArowB;
		//int ldb = colB,lda=colArowB,ldc=colB;
		//stat=cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, d_B, ldb, d_A, lda, &beta, d_C, ldc);
		Timer t;
		t.start();
		if(bColumnMajorFormat){
			//column major format
			int m=rowA, n=colB, k=colArowB;
			int lda = m,ldb=k,ldc=m;
			cublasStatus_t stat=cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, d_A, lda, d_B, ldb, &beta, d_C, ldc);
		}
		else {
			//row major format
			int m=colB, n=rowA, k=colArowB;
			int ldb = colB,lda=colArowB,ldc=colB;
			cublasStatus_t stat=cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alf, d_B, ldb, d_A, lda, &beta, d_C, ldc);
		}
		cudaMemcpy(C,d_C,sizeof(float)*sizeC,cudaMemcpyDeviceToHost); // copy device result to host 
		t.stop();
		std::cout<<"cublas "<<t.elapsedMilliseconds()<<std::endl;
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		return true;
}