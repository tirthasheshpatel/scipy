#include <stdio.h>
#include <stdlib.h>

typedef struct point_s {
  double x, y;
} point_t;


double segfault_null_pointer(point_t *point)
{
  return point->x + point->y;
}

int main()
{
    point_t *point = NULL;
    double res = segfault_null_pointer(point);
    printf("%lf\n", res);
}
