OBJECTS=hilbert_mapper.o
HEADERS=

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h $(HEADERS)
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

