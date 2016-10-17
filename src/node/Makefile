OBJECTS=branch.o \
				node.o \
				leaf_node.o \
				node_soa.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h 
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

branch.o : ./../common/macro.h ./../common/config.h
node.o : ./../common/macro.h ./../common/config.h
leaf_node.o : ./../common/macro.h ./../common/config.h
node_soa.o : ./../common/macro.h ./../common/config.h

clean:
	rm -f *.o

