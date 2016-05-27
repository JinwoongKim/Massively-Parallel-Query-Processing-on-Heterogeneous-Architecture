OBJECTS=chunk_manager.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

chunk_manager.o : ./../common/macro.h ./../common/config.h ./../common/logger.h

clean:
	rm -f *.o
