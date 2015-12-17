SUBDIRS=common \
				evaluator \
				node \
				mapper \
				sort \
				tree \
				io \
				main

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) 

$(SUBDIRS):
	$(MAKE) --directory $@

