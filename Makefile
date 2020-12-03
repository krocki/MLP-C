CC=gcc-10
LD=gcc-10

OPT_LEVEL=-mtune=native -march=native -Ofast -mavx2 -mfma

CC_OPTS=$(OPT_LEVEL) -fPIC -Wall -Wno-unused-variable -Werror -Wfatal-errors
LD_OPTS=$(OPT_LEVEL) -lm -lc -flto

AFFINITY=0

ifeq ($(AFFINITY), 1)
	LD_OPTS:=$(LD_OPTS) -lpthread
endif

HEADERS:=$(wildcard *.h) Makefile

.SUFFIXES:

TARGETS=softmax
all : $(TARGETS)

%.o: %.c $(HEADERS)
	$(CC) -c $< -o $@ $(CC_OPTS)

%: %.o rand.o
	$(LD) $^ -o $@ $(LD_OPTS)

clean:
	rm -rf $(TARGETS) *.o
