#ifndef __IO_H__
#define __IO_H__

#include <stdio.h>
#include <assert.h>

void store_f32(const char *fname, int len, float *data) {

  FILE *f = fopen(fname, "wb");
  assert(NULL != f);

  fwrite(data, sizeof(float), len, f);
  fclose(f);
}

int load(const char *fname, int offset, int size, unsigned char *data) {
  FILE *f;
  f = fopen(fname, "rb");
  if (f) {
    fseek(f, offset, SEEK_SET);
    if (!fread(data, size, 1, f)) {
      fprintf(stderr, "couldn't read %s\n", fname);
      return -1;
    }
    fclose(f);
    return 0;
  } else {
    fprintf(stderr, "couldn't open %s\n", fname);
    return -1;
  }
};

#endif // __IO_H__
