#include <stdlib.h>
#include <math.h>

#if defined(__x86_64__)
/*
   this way of getting random number is way better
   on a modern intel cpu compared to the standard rand()
   uses an instruction 'rdrand'
   undefine to fall back to the standard libc version
*/

/* return a rand float in range [0, 1) */
float randf() {

  unsigned long v;
  char c;
  do {
    __asm__ volatile(
        "rdrand %0; setc %1"
        : "=r" (v), "=qm" (c)
        );
  } while (c != 1);

  unsigned long ui_max = ~0;
  float f = (float)v / ((float)ui_max + 1.0f);
  return f;
}
#else
float randf() {
  return rand() / (RAND_MAX + 1.0f);
}
#endif
/* normal distribution, N(mean, std), n - number of values */
void randn(float *out, float mean, float std, int n) {
  for (int i=0; i<n; i++) {
    float  x = randf(),
           y = randf(),
           z = sqrtf(-2 * logf(x)) * cos(2 * M_PI * y);
    out[i] = std*z + mean;
  }
}

