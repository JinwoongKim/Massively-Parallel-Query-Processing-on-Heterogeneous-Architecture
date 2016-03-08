OBJECTS=thrust_sort.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

thrust_sort.o : ./../common/config.h ./../common/logger.h

clean:
	rm -f *.o

