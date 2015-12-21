SUBDIRS=common \
				evaluator \
				io \
				tree \
				node \
				sort \
				mapper \
				transformer \
				main

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) 

$(SUBDIRS):
	$(MAKE) --directory $@

