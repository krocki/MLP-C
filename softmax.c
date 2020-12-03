#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define echo(x) do { puts(#x); (x); } while (0)

void vec_print(int N, float *v) {
  for (int i=0; i<N; i++)
    printf("%7.3f%s", v[i], i<(N-1)? "" : "\n");
}
void digit_print(int N, float *v) {
  for (int i=0; i<N; i++)
    printf("%c%s", v[i]>0 ? '#' : ' ', (i%28)!=27? "" : "\n");
}

void forward0(int n_in, int n_out, float *in, float *out, float *w) {

  for (int i=0; i<n_in; i++)
    for (int j=0; j<n_out; j++)
      out[j] += w[j*n_in+i] * in[i];

}

void softmax0(int Y, float *y, float *p) {

  float m0 = .0f; /* find max */
  for (int k=0; k<Y; k++)
    m0 = (k>0 && y[k] > m0) ? y[k] : m0;

  float sum = .0f;

  for (int k=0; k<Y; k++) {
    p[k] = expf(y[k]-m0);
    sum += p[k];
  }

  for (int k=0; k<Y; k++) {
    p[k] /= sum;
  }

}

int argmax(int Y, float *p) {

  int idx = 0;
  float pmax = p[idx];

  for (int k=1; k<Y; k++) {
    if (pmax < p[k]) {
      pmax = p[k];
      idx = k;
    }
  }

  return idx;
}

#define X 784
#define Y 10
#define DATAPOINTS 50000

unsigned char inputs[X * DATAPOINTS];
unsigned char labels[DATAPOINTS];

#define SMOOTHING 0.9999f

extern void randn(float *out, float mean, float std, int n);

float cross_entropy(int N, float *p, float *t) {

  float ce = 0;
  for (int k=0; k<N; k++)
    ce += -logf(p[k]) * t[k];

  return ce;
}

float norm2(int N, float *x) {
  float n = .0f;
  for (int i=0; i<N; i++)
    n += x[i] * x[i];
  return n;
}

int main(int argc, char **argv) {

  if (0 > load("data/train-images-idx3-ubyte",
        16, X * DATAPOINTS, inputs)) return -1;
  if (0 > load("data/train-labels-idx1-ubyte",
        8, DATAPOINTS, labels)) return -1;

  float x[X];
  float y[Y];
  float t[Y];
  float p[Y];
  float w[X*Y];

  float dy[Y];
  float dw[X*Y];

  float decay = .0f;
  float lr = 1e-5f;

  int iters = 0;
  int max_iters = -1;

  float smooth_ce = logf(Y);
  float smooth_acc = 1.f/Y;

  // init w
  randn(w, 0, 0.001, X*Y);

  do {

    int r = random() % DATAPOINTS;

    memset(t, 0, sizeof(float) * Y);
    memset(y, 0, sizeof(float) * Y);
    memset(p, 0, sizeof(float) * Y);
    memset(x, 0, sizeof(float) * X);

    t[labels[r]] = 1.0f;
    for (int i=0; i < X; i++)
      x[i] = inputs[r*X+i] / 255.0f;

    /* y = w'x */
    forward0(X, Y, x, y, w);
    /* p := softmax(y) */
    softmax0(Y, y, p);

    //echo(vec_print(Y, y));
    //echo(vec_print(Y, p));
    int predicted = argmax(Y, p);
    float ce = cross_entropy(Y, p, t);
    smooth_ce = smooth_ce * SMOOTHING + (1.0f - SMOOTHING) * ce;
    smooth_acc = smooth_acc * SMOOTHING + (1.0f - SMOOTHING) * (predicted == labels[r]);

    if (0 == (iters % 50000)) {
      echo(digit_print(X, x));
      printf("r = %5d, predicted = %2d, true = %2d, ce = %f, (%f), norm w = %f, acc = %f\n", r, predicted, labels[r], ce, smooth_ce, norm2(X*Y, w), smooth_acc);
    }
    //echo(vec_print(X*Y, w));
    //echo(vec_print(Y, y));

    memset(dy, 0, sizeof(float) * Y);
    memset(dw, 0, sizeof(float) * Y * X);

    /* backprop */
    for (int k=0; k<Y; k++)
      dy[k] = p[k] - t[k];

    for (int j=0; j<X; j++)
      for (int k=0; k<Y; k++)
        dw[k*X+j] += x[j] * dy[k];

    /* adjust weights */
    for (int i=0; i<Y*X; i++)
      w[i] = w[i] * (1.0f - decay) - dw[i] * lr;

  } while (++iters < max_iters || -1 == max_iters);
}
