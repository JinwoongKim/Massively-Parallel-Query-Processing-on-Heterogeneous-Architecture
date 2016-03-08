OBJECTS=transformer.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h 
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

transformer.o : ./../common/config.h ./../common/macro.h ./../common/logger.h

clean:
	rm -f *.o

