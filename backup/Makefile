var = $1

all : rtree.cu main.cu
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lcudart main.cu
#nvcc -arch=sm_20 --ptxas-options=-v -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib -lmpi -lcudart main.cu //resource usage info.
2:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda2d${var}
4:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda4d${var}
8:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda8d${var}
16:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda16d${var}
32:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda32d${var}
64:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda64d${var}
128:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda128d${var}

256:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda256d${var}
512:
	nvcc -arch=sm_20 -o cuda -I${MPI_HOME}/include -L${MPI_HOME}/lib64 -lmpi -lcudart main.cu
	cp cuda cuda512d${var}
anal :
	nvcc -arch=sm_20 -o anal  analysis.cu 


#	nvcc -O3 -arch=sm_20 -Xptxas -dlcm=cg -o cuda main.cu  # //turn off l1 cache
#nvcc -DTHRUST_DEBUG -arch=sm_20 main.cu  //thrust debug mode
#	nvcc -O0 -Xcicc -O0 -Xptxas -O0 -arch=sm_20 -o cuda main.cu  // no optimization
