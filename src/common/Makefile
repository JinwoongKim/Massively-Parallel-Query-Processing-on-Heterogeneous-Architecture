OBJECTS=point.o evaluator.o
INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

