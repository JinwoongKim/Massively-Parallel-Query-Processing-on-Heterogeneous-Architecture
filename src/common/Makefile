OBJECTS=types.o
HEADERS=macro.h

INC=-I. -I../.

all: $(OBJECTS)

%.o: %.cpp %.h $(HEADERS)
	$(NVCC) -x cu $(NVCCFLAGS) $(INC) -dc $< -o $@ 

