SUBDIRS=common \
				evaluator \
				node \
				mapper \
				tree \
				io \
				main

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) 

$(SUBDIRS):
	$(MAKE) --directory $@

