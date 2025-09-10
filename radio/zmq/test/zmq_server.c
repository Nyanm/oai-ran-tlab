//  Hello World server
#include <zmq.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

void usage(void)
{
  printf("options:\n");
  printf("    -tx     run as TX server, not RX server\n");
  printf("    -a <address, like tcp://*:5555)\n");
  printf("    -c      connect, don't bind\n");
  exit(0);
}

int main(int n, char **v)
{
  char *a = 0;
  int tx = 0;
  int connect = 0;

  for (int i = 1; i < n; i++) {
    if (!strcmp(v[i], "-tx")) { tx = 1; continue; }
    if (!strcmp(v[i], "-c")) { connect = 1; continue; }
    if (!strcmp(v[i], "-a")) {if(i>n-2)usage(); a = v[++i]; continue; }
    usage();
  }
  if (!a) usage();

  //  Socket to talk to clients
  void *context = zmq_ctx_new();
  void *responder;
  if (tx)
    responder = zmq_socket(context, ZMQ_REP);
  else
    responder = zmq_socket(context, ZMQ_REQ);
  int rc;
  if (connect) {
    printf("connecting\n");
    rc = zmq_connect(responder, a);
  } else {
    printf("binding\n");
    rc = zmq_bind(responder, a);
  }
  if (rc != 0) { printf("bind/connect errpr\n"); exit(1); }

  if (tx)
    while (1) {
      char buffer [10];
      int l = zmq_recv(responder, buffer, 10, 0);
#define N (23040000 / 2000)
      printf("Received stuff l %d, sending 0 (%d)\n", l, N*4);
      char yo[4*N] = { 0 };
      zmq_send(responder, yo, 4*N, 0);
      usleep(100 * 1000);
    }
  else
    while (1) {
      char buffer [10];
      printf("start loop\n");
      zmq_send(responder, buffer, 1, 0);
      char yo[4*N*10] = { 0 };
      int l = zmq_recv(responder, yo, 10*4*N, 0);
      printf("Received stuff l %d\n", l);
      usleep(100 * 1000);
    }
  return 0;
}
