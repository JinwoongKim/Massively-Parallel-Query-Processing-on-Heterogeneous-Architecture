OBJECTS=tree.o \
        hybrid.o \
				mphr.o

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

tree.o : ./../common/macro.h ./../common/config.h ./../common/logger.h
hybrid.o : ./../common/macro.h ./../common/config.h ./../common/logger.h
mphr.o : ./../common/macro.h ./../common/config.h ./../common/logger.h

clean:
	rm -f *.o
