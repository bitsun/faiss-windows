to build faiss-test project in windows, pleas take the following steps.
1. download intel math kernel library ,set enviroment variable MKL_PATH to the directory containg the static libraries of intel MKL
2. install cuda 9.2,make sure enviroment variable CUDA_PATH is properly set
3. download boost, and set enviroment variable BOOST_DIR to the directory of boost 
4. open command line prompt,go to boost directory,build boost-test with the following switches:
   .\b2 toolset=msvc-14.0 variant=release link=static architecture=x86 address-model=64 define=BOOST_TEST_NO_MAIN --with-test --stagedir=.\staticlib\x64
5. We should now be able to build faiss-test project, and run some tests, for example to check the correctness of flat index implementation,run:
	faiss-test.exe  --run_test=FlatIndex/FP32FlatIndexCPU
