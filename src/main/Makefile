OBJECTS=main.o 
SRCS=main.cpp
INC=-I. -I../.

NVCC=nvcc
NVCCFLAGS=-arch=sm_20

all: $(OBJECTS)

%.o: %.cpp
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@

