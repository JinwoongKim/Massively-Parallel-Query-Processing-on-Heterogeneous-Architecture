OBJECTS=./src/*/*.o

NVCC=nvcc
NVCCFLAGS=-arch=sm_20

all:
	cd src; $(MAKE)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -o ./bin/cuda

clean:
	find . -type f -name "*.o" -delete; find ./bin/ -type f -name "cuda" -delete
