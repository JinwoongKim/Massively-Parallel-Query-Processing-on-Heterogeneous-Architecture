OBJECTS= mphr.o
HEADERS= tree.h mphr.h

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp $(HEADERS)
	rm -f *.o
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

