#include <stdlib.h>
#include <math.h>

#if defined(__x86_64__)
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
void randn(float *out, float mean, float std, int n) {
  for (int i=0; i<n; i++) {
    float  x = randf(),
           y = randf(),
           z = sqrtf(-2 * logf(x)) * cos(2 * M_PI * y);
    out[i] = std*z + mean;
  }
}

