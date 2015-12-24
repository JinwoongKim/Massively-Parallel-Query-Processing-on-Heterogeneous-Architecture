OBJECTS=branch.o \
				node.o \
				g_node.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h ../common/config.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

clean:
	rm -f *.o

