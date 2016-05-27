SUBDIRS=common \
				evaluator \
				io \
				tree \
				node \
				sort \
				mapper \
				transformer \
				manager \
				main

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) 

$(SUBDIRS):
	$(MAKE) --directory $@

