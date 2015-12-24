OBJECTS=evaluator.o \
				recorder.o

INC=-I. -I../.

all: $(OBJECTS) $(HEADERS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

clean:
	rm -f *.o
