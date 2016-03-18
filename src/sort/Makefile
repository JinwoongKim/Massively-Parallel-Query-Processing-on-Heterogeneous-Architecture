OBJECTS=thrust_sorter.o\
        parallel_sorter.o\
        sorter.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

thrust_sorter.o : ./../common/config.h ./../common/logger.h
parallel_sorter.o : ./../common/logger.h
sorter.o : ./../common/logger.h

clean:
	rm -f *.o

