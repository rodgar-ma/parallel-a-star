#include <stdlib.h>
#include <stdio.h>
#include "priority_list.h"
#include "astar.h"

void print_list(priority_list *list) {
    for (size_t i = 0; i < list->count; i++)
    {
        printf("Node %ld, fCost=%f\n", list->nodes[i]->id, list->nodes[i]->fCost);
    }
}

int main(int argc, char const *argv[])
{
    priority_list *list = priority_list_create();

    node *n0 = node_create(0, 0, 5.99999, NULL);
    node *n1 = node_create(1, 1, 4.9999, NULL);
    node *n2 = node_create(2, 2, 6, NULL);
    node *n3 = node_create(3, 2, 2.123232, NULL);
    node *n4 = node_create(4, 2, 0, NULL);
    node *n5 = node_create(5, 3, 1, NULL);
    node *n6 = node_create(6, 3, 9, NULL);

    priority_list_insert_or_update(list, n0);
    priority_list_insert_or_update(list, n1);
    priority_list_insert_or_update(list, n2);
    priority_list_insert_or_update(list, n3);
    priority_list_insert_or_update(list, n4);
    priority_list_insert_or_update(list, n5);
    priority_list_insert_or_update(list, n6);

    print_list(list);
    printf("\n");
    node *first = priority_list_extract(list);
    print_list(list);
    printf("\n");
    first = priority_list_extract(list);
    print_list(list);
    printf("\n");
    first = priority_list_extract(list);
    print_list(list);
    printf("\n");
    first = priority_list_extract(list);
    print_list(list);

    return 0;
}
