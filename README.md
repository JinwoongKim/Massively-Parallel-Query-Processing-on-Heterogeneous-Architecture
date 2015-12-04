In order to make everything work, once you clone this directory, go the root of project
, and type:
> aclocal

> autoconf

> automake --add-missing

> ./bootstrap

This should create all "configure" script along with all the Makefile's. Now, you can t
ype
> ./configure

> make && make install

And your project should compile. Now you are all set with a lean template to build upon
.
