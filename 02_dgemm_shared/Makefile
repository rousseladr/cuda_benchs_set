CC=nvcc
CFLAGS=-O3
LDFLAGS=

EXE=dgemm_shared.exe

all : $(EXE)

%.o : %.cu
	$(CC) -c $(CFLAGS) -o $@ $^

../common/common.o : ../common/common.c
	$(CC) -c $(CFLAGS) -o $@ $^

$(EXE) : dgemm_shared.o ../common/common.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

proper :
	rm -f *.o ../common/*.o

clean : proper
	rm -f $(EXE)
