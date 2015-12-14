SUBDIRS=common \
				evaluator \
				tree \
				io \
				main

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS) 

$(SUBDIRS):
	$(MAKE) --directory $@

