export NVCC=nvcc
export NVCCFLAGS= -default-stream per-thread -arch=sm_35 -std=c++11 -w -ltbb $(OPTION)
# (by_passing) cache off : -Xptxas -dlcm=cg
#export NVCCFLAGS= -default-stream per-thread -arch=sm_35 -Xptxas -dlcm=cg -std=c++11 -w -ltbb $(OPTION)

OBJECTS=./src/*/*.o

all: 
	cd src; $(MAKE)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o ./bin/cuda

debug:
	find . -type f -name "*.o" -delete; find ./bin/ -type f -name "cuda" -delete
	cd src; $(MAKE)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o ./bin/cuda

clean:
	find . -type f -name "*.o" -delete; find ./bin/ -type f -name "cuda" -delete
