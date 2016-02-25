CC=g++
INCLUDES=-I../eigen
CFLAGS=-Wall -std=c++11
SOURCES=random_matrix_factory.cc random_orthonormal_basis.cc random_svd.cc

all: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) $(SOURCES)

rsvd: random_svd.cc random.

rob: random_orthonormal_basis.cc random_orthonormal_basis.h random_matrix_factory.o
	$(CC) $(CFLAGS) random_matrix_factory.o random_orthonormal_basis.cc

rmf: random_matrix_factory.cc random_matrix_factory.h
	$(CC) $(CFLAGS) $(INCLUDES) -fPIC -c random_matrix_factory.cc -o random_matrix_factory.o
	$(CC) $(CFLAGS) $(INCLUDES) -shared -Wl,-soname,librmf.so.1 -o librmf.so.1.0 random_matrix_factory.o
	mv librmf.so.1.0 ../shared
	ln -sf ../shared/librmf.so.1.0 ../shared/librmf.so.1
	ln -sf ../shared/librmf.so.1.0 ../shared/librmf.so

clean:
	rm *o
