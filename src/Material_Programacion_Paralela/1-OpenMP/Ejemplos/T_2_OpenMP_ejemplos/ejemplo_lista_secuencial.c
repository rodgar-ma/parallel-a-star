#include <omp.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int arr[] = {0, 1, 2, 3};
    int *ptr = arr;
    for (int i = 0; i < 4; i++)
    {
        printf("%d\n", *ptr);
        ptr++;
    }
    
    return 0;
}
