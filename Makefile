CC=gcc
LD=gcc

OPT_LEVEL=-Ofast

CC_OPTS=$(OPT_LEVEL) -fPIC -Wall -Wno-unused-variable -Werror -Wfatal-errors

HEADERS:=$(wildcard *.h) Makefile

.SUFFIXES:

TARGETS=mlp
all : $(TARGETS)

%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@ $(CC_OPTS)

%: %.o rand.o
	$(CC) $^ -o $@ $(CC_OPTS) -lm

clean:
	rm -rf $(TARGETS) *.o
