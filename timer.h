#include "rdtsc.h"
#include <sys/time.h>

double get_time() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

#define timeit(expr)  \
  do { double ts=get_time(); \
       unsigned long long cs=rdtsc(); \
       (expr); \
       unsigned long long ce=rdtsc(); \
       double te=get_time(); \
       printf("`" #expr "` took %f s, %llu cycles\n",\
           te-ts, ce-cs); \
  } while (0)

#define timeit2(expr, val)  \
  do { double ts=get_time(); \
       unsigned long long cs=rdtsc(); \
       (expr); \
       unsigned long long ce=rdtsc(); \
       double te=get_time(); \
       *(val) = te-ts; \
  } while (0)
