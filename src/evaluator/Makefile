OBJECTS=evaluator.o \
				recorder.o

INC=-I. -I../.

all: $(OBJECTS) $(HEADERS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

evaluator.o : ./../common/config.h ./../common/macro.h ./../common/logger.h 

clean:
	rm -f *.o
