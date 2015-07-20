INCLUDES=-I${MPI_HOME}/include -I${PWD}/header/
all : 
	nvcc -arch=sm_20 -Xptxas -g -G -rdc=true -o ./bin/cuda $(INCLUDES) main.cu src/*
#nvcc -arch=sm_20 -Xptxas -g -G -o cuda $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu 
#	--ptxas-options=-v

cache_on : 
	nvcc -arch=sm_20 -Xptxas -dlcm=ca -o cuda_on $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu 
cache_off : 
	nvcc -arch=sm_20 -Xptxas -dlcm=cg -o cuda_off $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu 
2:
	nvcc -arch=sm_20 -o cuda2d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
3:
	nvcc -arch=sm_20 -o cuda3d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
4:
	nvcc -arch=sm_20 -o cuda4d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
8:
	nvcc -arch=sm_20 -o cuda8d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
16:
	nvcc -arch=sm_20 -o cuda16d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
32:
	nvcc -arch=sm_20 -o cuda32d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
64:
	nvcc -arch=sm_20 -o cuda64d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
128:
	nvcc -arch=sm_20 -o cuda128d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
256:
	nvcc -arch=sm_20 -o cuda256d $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu
512:
	nvcc -arch=sm_20 -o cuda512 $(INCLUDES) -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu

#nvcc  -arch=sm_20 -Xptxas -dlcm=ca -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu 
#nvcc  -arch=sm_20 -g -G -Xptxas -v -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lcudart -lineinfo main.cu 
	
