#include "stdlib.h"
#include "stdio.h"

typedef struct __MyNode MyNode;

struct __MyNode {
  int x;
  int y;
  int price;
};

int main(int argc, char const *argv[])
{
  MyNode **nodes = malloc(10 * sizeof(MyNode*));
  for (int i = 0; i < 10; i++) {
    nodes[i] = malloc(10 * sizeof(MyNode));
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      MyNode n;
      n.x = i;
      n.y = j;
      n.price = i*j;
      nodes[i][j] = n;
    }
  }

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      printf("node[%d][%d]\n", nodes[i][j].x, nodes[i][j].y);
    }
  }

  printf("%lu bytes\n", sizeof(nodes));
  printf("%lu bytes\n", sizeof(MyNode));
  printf("%lu bytes\n", sizeof(nodes[0][0]));
  
  return 0;
}
