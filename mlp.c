#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>

/* affinity - may not work without GNU_SOURCE*/
#include <sched.h> 
cpu_set_t  mask;
void set_affinity(int core_id)
{
  printf("setting cpu affinity -> %d\n", core_id);
  CPU_ZERO(&mask);
  CPU_SET(core_id, &mask);
  sched_setaffinity(0, sizeof(mask), &mask);
}
/* ******************* */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* batch size */
#define B 8
/* input size */
#define X 784
/* hidden size */
#define H 100
/* output size */
#define Y 10


#define ITERATIONS 1000000
#define TARGET_ACC 0.965f
#define STATS_INTERVAL 50000
#define SMOOTHING 0.99999f
#define LEARNING_RATE 5 * 1e-4f
#define DATAPOINTS 50000
#define WEIGHT_DECAY .0f
#define LOGISTIC 1
#define DROPOUT 0.0
#define RELU 0
#define TANH 0
//#define RHO_TARGET 0.2f
//#define BETA 1e-5f

extern void randn(float *out, float mean, float std, int n);

unsigned char inputs[X * DATAPOINTS];
unsigned char labels[DATAPOINTS];

double get_time() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
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

int main(int argc, char **argv) {

  if (argc > 1)
    if (0 == strcmp(argv[1], "help")) {
      printf("usage: %s cpu_core max_iters lr decay\n", argv[0]);
      return 0;
    }

  /* x -w-> h -v-> y */
  float *x,*h,*y,*p,*t,*c; /*states*/
  float *w,*v;             /*weights*/
  float *dh,*dy;           /*states-grads*/
  float *dw,*dv;           /*weight-grads*/
  float *m;                /*dropout*/

  x  = (float *) malloc( sizeof(float) * X * B);
  w  = (float *) malloc( sizeof(float) * X * H);
  dw = (float *) malloc( sizeof(float) * X * H);
  h  = (float *) malloc( sizeof(float) * H * B);
  dh = (float *) malloc( sizeof(float) * H * B);
  m  = (float *) malloc( sizeof(float) * H * B);
  v  = (float *) malloc( sizeof(float) * H * Y);
  dv = (float *) malloc( sizeof(float) * Y * H);
  dy = (float *) malloc( sizeof(float) * Y * B);
  y  = (float *) malloc( sizeof(float) * Y * B);
  p  = (float *) malloc( sizeof(float) * Y * B);
  c  = (float *) malloc( sizeof(float) * Y * B);
  t  = (float *) malloc( sizeof(float) * Y * B);

  float smooth_act = 0.0f;
  float smooth_ce = logf(Y);
  float smooth_acc = 1.0f/Y;

  int cpu_core =  argc > 1 ? atoi(argv[1]) : -1;
  int max_iters = argc > 2 ? atoi(argv[2]) : ITERATIONS;
  float lr =      argc > 3 ? atof(argv[3]) : LEARNING_RATE;
  float decay =   argc > 4 ? atof(argv[4]) : WEIGHT_DECAY;

  if (cpu_core >= 0) set_affinity(cpu_core);

  if (0 > load("data/train-images-idx3-ubyte",
        16, X * DATAPOINTS, inputs)) return -1;
  if (0 > load("data/train-labels-idx1-ubyte",
        8, DATAPOINTS, labels)) return -1;

  /* init weights */
  randn(w, .0f, 0.1f, X * H);
  randn(v, .0f, 0.1f, Y * H);

  double gflops_per_sample =
    (double)(2 * (X*H+H*Y) * 2) /
    ( 1 << 30 );

  int samples=0, iters=0;
  srand(33);

  double t0 = get_time();
  double start_time = t0;

  do {

    /* random sample */
    int r[B];

    for (int b=0; b < B; b++)
      r[b] = random() % DATAPOINTS;

    memset(t, 0, sizeof(float) * Y * B);
    memset(h, 0, sizeof(float) * H * B);
    memset(y, 0, sizeof(float) * Y * B);

    for (int b=0; b < B; b++) {
      t[b*Y+labels[r[b]]] = 1.0f;
      for (int i=0; i < X; i++)
        x[b*X+i] = inputs[r[b]*X+i] / 255.0f;
    }

    /* h := w'x */
    /* col major */
    /* h [H rows, B cols] */
    /* w [X rows, H cols] */
    /* x [X rows, B cols] */
    for (int j=0; j<H; j++)
    for (int i=0; i<X; i++)
    for (int b=0; b<B; b++)
        h[b*H+j] +=
        w[j*X+i] * x[b*X+i];

    /* nonlinearity */
    for (int j=0; j<H*B; j++)
    #if LOGISTIC
      h[j] = 1.0f / (1.0f + expf(-h[j]));
    #endif
    #if RELU
      h[j] = h[j] < 0.0f ? 0.0f : h[j];
    #endif
    #if TANH
      h[j] = tanhf(h[j]);
    #endif

    if (DROPOUT > 0) {
      for (int j=0; j<H*B; j++) {
        m[j] = ((float)random() / (float)RAND_MAX) < DROPOUT ? 0.0f : 1.0f;
        h[j] *= m[j];
      }
    }

    float act_sum = 0.0f;
    for (int i=0; i<H*B; i++)
      act_sum += h[i];

    smooth_act = SMOOTHING * smooth_act
        + (1.0f - SMOOTHING) * act_sum/(H*B);

    /* y := vh */
    for (int b=0; b<B; b++)
    for (int j=0; j<H; j++)
    for (int k=0; k<Y; k++)
      y[b*Y+k] += v[k*H+j] * h[b*H+j];

    /* p := softmax(y) */
    for (int b=0; b<B; b++) {
      float m0 =.0f; /* find max */
      for (int k=0; k<Y; k++)
        m0 = k>0 && (y[b*Y+k] > m0) ?
        y[b*Y+k] : m0;

      float sum = .0f;

      for (int k=0; k<Y; k++) {
        p[b*Y+k] = expf(y[b*Y+k]-m0);
        sum += p[b*Y+k];
      }

      for (int k=0; k<Y; k++)
        p[b*Y+k] /= sum;
    }

    /* forward pass end */
    /* bookkeeping for stats */
    int argmax[B];
    float probmax[B];
    for (int b=0; b<B; b++) {
      argmax[b] = -1;
      probmax[b] = .0f;
      for (int k=0; k<Y; k++) {
        if (probmax[b] < p[b*Y+k] || k==0) {
          probmax[b] = p[b*Y+k];
          argmax[b] = k;
        }
        c[b*Y+k] = -logf(p[b*Y+k]) * t[b*Y+k];
        smooth_ce = smooth_ce * SMOOTHING +
        (1.0f-SMOOTHING) * c[b*Y+k];
      }
      smooth_acc = smooth_acc * SMOOTHING +
      (1.0f-SMOOTHING) * (argmax[b]==labels[r[b]]);
    }

    if (0 == (samples % STATS_INTERVAL) &&
      samples > 0) {
      float time_d = get_time() - t0;
      float samples_per_sec = STATS_INTERVAL / time_d;
      float gflops_per_sec = samples_per_sec *
        gflops_per_sample;
        printf("[%4.3f s] "
             "acc=%3.2f%%, "
             "ce=%3.3f, "
             "%.2f samples/sec, "
             "%.2f gflop/s\n",
          get_time() - start_time, 100.0*smooth_acc, smooth_ce,
          samples_per_sec, gflops_per_sec);

      t0 = get_time();
    }

    /* backprop begin */
    /* reset grads */
    memset(dh, 0, sizeof(float) * H * B);
    memset(dw, 0, sizeof(float) * H * X);
    memset(dv, 0, sizeof(float) * H * Y);

    /* dy */
    for (int b=0; b<B; b++)
    for (int k=0; k<Y; k++)
      dy[b*Y+k] = p[b*Y+k] - t[b*Y+k];

    /* dv := h * dy' */
    for (int b=0; b<B; b++)
    for (int j=0; j<H; j++)
    for (int k=0; k<Y; k++)
      dv[k*H+j] += h[b*H+j] * dy[b*Y+k];

    /* dh := v * dy */
    for (int b=0; b<B; b++)
    for (int j=0; j<H; j++)
    for (int k=0; k<Y; k++)
      dh[b*H+j] += v[k*H+j] * dy[b*Y+k];

    /* sparsity */
    /* see A.Ng's Sparse Autoencoder */
    /* https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf */

    /*
    for (int b=0; b<B; b++)
      for (int j=0; j<H; j++)
        dh[j] -= BETA *
          ( -act_sum/RHO_TARGET +
            (1.0f - act_sum) / (1.0f - RHO_TARGET)
          );
    */

    /* nonlinearity on h */
    for (int j=0; j<H*B; j++)
    #if LOGISTIC
      dh[j] = dh[j] * h[j] * (1.0f - h[j]);
    #endif
    #if RELU
      dh[j] = dh[j] * h[j];
    #endif
    #if TANH
      dh[j] = dh[j] * (1.0f - h[j] * h[j]);
    #endif

    /* dw := x * dh' */
    for (int j=0; j<H; j++)
    for (int i=0; i<X; i++)
    for (int b=0; b<B; b++)
      dw[j*X+i] += x[b*X+i] * dh[b*H+j];
    /* backprop end */

    /* adjust weights */
    for (int i=0; i<H*X; i++)
      w[i] = w[i] * (1.0f - decay) - dw[i] * lr;
    for (int i=0; i<H*Y; i++)
      v[i] = v[i] * (1.0f - decay) - dv[i] * lr;

    samples+=B;

  } while (iters++ < max_iters && smooth_acc < TARGET_ACC);

  free(x), free(w), free(dw);
  free(h), free(dh);
  free(m);
  free(v), free(dv);
  free(y), free(dy);
  free(p), free(c), free(t);

  return 0;
}
