OBJECTS=hilbert_mapper.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

hilbert_mapper.o : ./../common/macro.h hilbert_macro.h

