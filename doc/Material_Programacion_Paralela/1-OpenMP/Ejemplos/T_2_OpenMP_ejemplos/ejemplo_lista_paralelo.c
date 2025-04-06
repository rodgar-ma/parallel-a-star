#include <omp.h>
#include <stdio.h>

int main(int argc, char const *argv[])
{
    int arr[] = {10, 20, 30};
    int *ptr=arr;
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            for (int i = 0; i < 4; i++)
            {
                #pragma omp task firstprivate (ptr)
                {
                    printf("%d\n", *ptr);
                    ptr++;
                }
            } 
        }
        printf("Hi\n");
    }
    
    return 0;
}
