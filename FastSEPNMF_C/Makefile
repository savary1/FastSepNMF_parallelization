CC = gcc
CFLAGS = -I.
DEPS = ReadWrite.h
LIBS = -lm

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

FastSEPNMF: FastSEPNMF.o ReadWrite.o
	$(CC) -o FastSEPNMF FastSEPNMF.o ReadWrite.o $(LIBS)

clean:
	rm -f FastSEPNMF FastSEPNMF.o ReadWrite.o