OBJECTS=rect.o \
				branch.o
HEADERS=node.h

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h $(HEADERS)
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

clean:
	rm -f *.o

