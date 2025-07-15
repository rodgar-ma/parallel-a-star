# Estrategias de paralelización del algoritmo A* en entornos de memoria compartida
## Trabajo de Fin de Grado.

En este repositorio se encuentra el código y la documentación del Trabajo de Fin de Grado (TFG) en Ingeniería Informática realizado por Miguel Ángel Rodríguez García en 2025 en la Universidad de Murcia.

El trabajo consiste en un estudio del Algoritmo A* (A-star) y las distintas estrategias de paralelización del mismo que han surgido en las últimas décadas. Comenzamos introduciendo los coceptos matemáticos necesarios, así como el problema que trata de resolver este algoritmo: el problema el camino más corto. Se introducen los algoritmos de búsqueda que han surgido historicamente para afrontar este problema y mostramos cómo llevaron la concepción de A*, cuyo principal hito fue introducir la información heurísitca en las estrategias de búsqueda [[1]](#1).

## Código

En cuanto al código que se puede encontrar en el repositorio, se trata de una serie de implementaciones del algoritmo A* bajo una misma interfaz. Entre los algoritmos se incluyen:

- Una versión secuencial de A*.
- Una versión paralela centralizada basada en SPA*(*Simple Parallel A\**)
- Una versión paralela distribuida basada en HDA* (*Hash Distributed A\**) [[2]](#2)

Todas las versiones están han implementado con el lenguaje C y usando OpenMP para la gestión de los hilos.

Para poder usar las implementaciones se debe definir una estructura `AStarSource` y se deben definir dos métodos:

- `void (*get_neighbors)(neighbors_list *neighbors, int n_id)`. Esta función es llamada por el algoritmo para obtener los vecinos del nodo `n_id`. Se deben añadir los vecinos a la lista `neighbors` utilizando el método `void add_neighbor(neighbors_list *neighbors, int n_id, float cost)`.
- `float (*heuristic)(int n1_id, int n2_id)`. Esta función debe implementar el cálculo del valor heurístico para dos nodos.

Para ejecutar los algoritmos cada uno de los algoritmos tenemos las siguientes opciones:

```{bash}
./main seq <scen_file.scen>             # Algoritmo secuencial (A*)
./main spa n_threads <scen_file.scen>   # Algoritmo centralizado (SPA*)
./main seq n_threads <scen_file.scen>   # Algoritmo descentralizado (HDA*)
```

## Referencias
<a id="1">[1]</a>
Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE transactions on Systems Science and Cybernetics, 4(2), 100-107.


<a id="2">[2]</a>
Kishimoto, A., Fukunaga, A., & Botea, A. (2009, October). Scalable, parallel best-first search for optimal sequential planning. In Proceedings of the International Conference on Automated Planning and Scheduling (Vol. 19, pp. 201-208).