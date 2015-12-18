OBJECTS=types.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h 
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

