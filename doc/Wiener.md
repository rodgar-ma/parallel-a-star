Sobre un Problema de la Geometría del Sitio
Por Carl Hierholzer en Karlsruhe

El problema que se resolverá a continuación, y que, hasta donde sé, no ha sido tratado antes, consiste en describir un método para encontrar la salida de un laberinto.

Un laberinto es un camino continuo y entrelazado con límites infranqueables, situado en un espacio cerrado, del cual solo puede salirse por una o pocas aberturas. Dado que el camino siempre mantiene un ancho determinado, su borde no puede tener puntos dobles, ni siquiera en los cruces de caminos. Por lo tanto, salvo por la dirección, un borde del camino solo puede recorrerse de una manera específica; en ningún punto existe la posibilidad de avanzar en múltiples direcciones.

El borde de un camino, al ser recorrido, puede llevar de vuelta a un punto ya visitado; en ese caso, debe incorporarse a la trayectoria anterior, ya que solo hay una posibilidad de avanzar, formando así una línea cerrada. Por otro lado, si no regresa sobre sí mismo, debido a su longitud finita, no puede quedar completamente encerrado en su entorno, por lo que necesariamente conduce hacia el exterior. Esto sucede en ambas direcciones en las que se puede recorrer, lo que implica que el borde termina en dos extremos abiertos hacia el exterior. Si el laberinto tiene solo una salida, los dos lados del mismo formarán esas dos terminaciones, y todas las demás líneas de borde serán cerradas.

De ello se deriva la siguiente regla: al entrar en el laberinto, se debe elegir una de las líneas de borde que conducen al exterior y seguirla hacia el interior; esta misma línea debe necesariamente volver a conducir al exterior.

Si uno se encuentra dentro del laberinto sin haber seguido una línea de borde desde el exterior, aún es posible encontrar la salida, siempre que se pueda recordar el camino recorrido y la dirección en la que se ha avanzado, o se pueda marcar de alguna manera.

Para ello, es más conveniente seguir el eje del camino en lugar de su borde. Esta posibilidad se basa en el hecho de que, mientras no se haya alcanzado la salida, una parte ya recorrida del eje del camino debe necesariamente cruzarse con una sección aún no recorrida. De lo contrario, dicha parte estaría completamente cerrada en sí misma y no estaría conectada con el camino de entrada.

Por lo tanto, se debe marcar el camino recorrido junto con la dirección en la que se avanza. Si se llega a un punto ya marcado, se debe retroceder y recorrer el camino marcado en sentido inverso. Si no se toma ningún desvío, este proceso llevaría a recorrer nuevamente toda la trayectoria hasta encontrar un nuevo camino aún no marcado, que entonces se debe seguir hasta toparse con otro camino marcado. En este punto, se repite el procedimiento anterior.

De este modo, siempre se añadirán nuevas secciones del camino a las ya recorridas, lo que eventualmente permitirá atravesar todo el laberinto y, en cualquier caso, encontrar la salida, si no se ha alcanzado antes.

Karlsruhe, diciembre de 1871

Sobre la Posibilidad de Recorrer un Camino sin Repetición ni Interrupción
Por Carl Hierholzer

Comunicado por Carl Wiener

En un trazado lineal arbitrario, se pueden definir ramas de un punto como las diferentes partes del camino por las que se puede salir desde ese punto. Un punto con múltiples ramas se denomina nodo y su grado corresponde al número de ramas que posee. Un nodo puede ser par o impar dependiendo del número de conexiones.

Si un trazado lineal puede recorrerse en un solo trazo sin repetir ningún segmento, entonces tiene exactamente cero o dos nodos impares. Cada vez que se atraviesa un punto en el recorrido, se emplean dos ramas. Por lo tanto, cualquier punto que se cruce un número impar de veces debe tener un número par de ramas. Un nodo impar solo puede ser el inicio o el final del recorrido.

Si un trazado tiene exactamente dos nodos impares, entonces debe comenzarse en uno de ellos y necesariamente terminar en el otro. Si no tiene nodos impares, el trazado debe ser un ciclo cerrado.

Esta demostración también implica que cualquier trazado lineal solo puede tener un número par de nodos impares.

(*) La siguiente investigación fue presentada por el fallecido Dr. Hierholzer, quien, lamentablemente, fue arrebatado prematuramente del servicio a la ciencia por la muerte († 13 de septiembre de 1871), ante un círculo de matemáticos amigos. Para evitar que cayera en el olvido, y debido a la falta de registros escritos, fue reconstruida de memoria con la ayuda de mi estimado colega Luroth, tratando de ser lo más fiel posible en la siguiente exposición.

Nota del editor: La idea esencial de este estudio, aunque presentada de manera más concisa y sin una demostración detallada, ya aparece en el tratado de Listing "Vorstudien zur Topologie", publicado en Göttinger Studien en 1847. Quizás este artículo ayude a atraer nuevamente la atención de los geómetras hacia este trabajo valioso en múltiples aspectos.

