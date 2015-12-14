OBJECTS=dataset.o
HEADERS=dataset.h

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp $(HEADERS)
	rm -f *.o
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

