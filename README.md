Motor de Inferencia por Enumeración con Redes Bayesianas

Proyecto para implementar un motor de inferencia por enumeración sobre una Red Bayesiana, usando Python y orientación a objetos, tal como se explicó en clase de Métodos Probabilísticos / Inteligencia Artificial.

El proyecto incluye:
	1.	Estructura de red bayesiana (nodos, padres, tablas de probabilidad).
	2.	Motor de inferencia por enumeración (algoritmo enumeration-ask).
	3.	Ejemplo de validación con la red del tren / cita (la misma vista en clase / presentación de la profesora).

⸻

1. Requisitos
	•	Python 3.8+ (recomendado 3.10 o superior)
	•	Sistema operativo: Windows, Linux o macOS.
	•	Consola / terminal para ejecutar el programa.

No se requieren librerías externas: solo se usan módulos estándar de Python (dataclasses, typing, argparse, etc.).

⸻

2. Estructura del proyecto

Archivos principales:

.
├── proyecto3.py           # Código fuente principal del motor de inferencia
├── estructura_red.txt     # Archivo con la estructura de la Red Bayesiana
└── cpts_red.txt           # Archivo con las tablas de probabilidad (CPTs)


⸻

3. Archivo de estructura: estructura_red.txt

Este archivo define la estructura de la red en términos de dependencias entre nodos.
	•	Una relación Padre → Hijo por línea.
	•	Formato:

Padre -> Hijo


	•	Se permiten:
	•	Líneas en blanco
	•	Comentarios que empiezan con #

Ejemplo (red del tren / cita)

# Estructura Red Bayesiana - Ejemplo tren / cita
Rain -> Maintenance
Rain -> Train
Maintenance -> Train
Train -> Appointment

Interpretación:
	•	Maintenance depende de Rain.
	•	Train depende de Rain y de Maintenance.
	•	Appointment depende de Train.

⸻

4. Archivo de CPTs: cpts_red.txt

Este archivo define las tablas de probabilidad de cada nodo (CPT: Conditional Probability Table).

Formato general por nodo:

NODE <NombreNodo>
VALUES v1 v2 v3 ...
[PARENTS padre1 padre2 ...]   # opcional si no tiene padres
TABLE
# si NO tiene padres:
p(v1) p(v2) p(v3) ...
# si tiene padres:
val_padre1 val_padre2 ... p(v1) p(v2) p(v3) ...
...
ENDNODE

Reglas:
	•	Las probabilidades de cada fila deben sumar 1.
	•	El orden de los valores en VALUES define el orden de las probabilidades en cada fila.
	•	Los nombres de nodos y valores son cadenas sin espacios (puedes usar _ si necesitas).

Ejemplo completo (tren / cita)

# CPTs para ejemplo del tren / cita

# Nodo Rain: {none, light, heavy}
NODE Rain
VALUES none light heavy
TABLE
0.7 0.2 0.1
ENDNODE

# Nodo Maintenance: depende de Rain
# R   yes   no
NODE Maintenance
VALUES yes no
PARENTS Rain
TABLE
none 0.4 0.6
light 0.2 0.8
heavy 0.1 0.9
ENDNODE

# Nodo Train: depende de Rain y Maintenance
# R     M    on_time  delayed
NODE Train
VALUES on_time delayed
PARENTS Rain Maintenance
TABLE
none  yes   0.8  0.2
none  no    0.9  0.1
light yes   0.6  0.4
light no    0.7  0.3
heavy yes   0.4  0.6
heavy no    0.5  0.5
ENDNODE

# Nodo Appointment: depende de Train
# T        attend  miss
NODE Appointment
VALUES attend miss
PARENTS Train
TABLE
on_time 0.9 0.1
delayed 0.6 0.4
ENDNODE

Este ejemplo es el mismo de la presentación: red de lluvia (Rain), mantenimiento (Maintenance), tren (Train) y cita (Appointment).

⸻

5. Código principal: proyecto3.py

El archivo proyecto3.py contiene:
	•	Clase Node: representa un nodo de la red (nombre, valores, padres, CPT).
	•	Clase BayesianNetwork:
	•	Manejo de estructura (agregar aristas, nodos, padres, hijos).
	•	Carga de red desde archivos (from_files).
	•	Impresión de estructura (print_structure).
	•	Impresión de CPTs (print_cpts).
	•	Inferencia por enumeración (enumeration_ask).
	•	Función main(): maneja argumentos por línea de comandos y ejecuta la inferencia solicitada.

El código está preparado para:
	•	Mostrar la estructura de la red.
	•	Mostrar las tablas de probabilidad.
	•	Hacer inferencia con evidencia.
	•	Mostrar una traza detallada del algoritmo si se usa --verbose.

⸻

6. Cómo ejecutar el proyecto

6.1. Desde línea de comandos

Ubícate en la carpeta donde están los archivos:

cd ruta/del/proyecto

Comando base

python proyecto3.py --estructura estructura_red.txt --cpt cpts_red.txt

Esto:
	•	Carga la red.
	•	Imprime:
	•	Estructura (nodos, padres, hijos).
	•	CPTs de cada nodo.
	•	No realiza inferencia porque no se especifica consulta.

En Windows, si python no funciona, prueba con:

py proyecto3.py --estructura estructura_red.txt --cpt cpts_red.txt



⸻

6.2. Hacer una consulta de inferencia

Usa la opción --consulta (-q) para indicar la variable a consultar, y --evidencia (-e) para las evidencias:
	•	Formato de evidencia: Var=valor (sin espacios).
	•	Puedes pasar varias evidencias: --evidencia Var1=val1 Var2=val2 ...

Ejemplo 1: P(Train | Rain=heavy)

python proyecto3.py \
  --estructura estructura_red.txt \
  --cpt cpts_red.txt \
  --consulta Train \
  --evidencia Rain=heavy

Salida esperada (formato):

Realizando inferencia para Train dado evidencia {'Rain': 'heavy'}
Distribución de probabilidad de Train dado la evidencia:
  P(Train=on_time | evidencia) = ...
  P(Train=delayed | evidencia) = ...


⸻

Ejemplo 2: P(Appointment | Rain=light, Maintenance=no)

python proyecto3.py \
  --estructura estructura_red.txt \
  --cpt cpts_red.txt \
  --consulta Appointment \
  --evidencia Rain=light Maintenance=no


⸻

6.3. Ver traza del algoritmo (modo verbose)

Si quieres mostrar en clase el paso a paso de la enumeración (traza), añade --verbose:

python proyecto3.py \
  --estructura estructura_red.txt \
  --cpt cpts_red.txt \
  --consulta Appointment \
  --evidencia Rain=light Maintenance=no \
  --verbose

El programa mostrará:
	•	Para cada variable:
	•	Si está en la evidencia: usa directamente P(Y = y | padres).
	•	Si no está en la evidencia: suma sobre todos sus valores posibles.
	•	Subtotales por rama de enumeración.
	•	Resultado sin normalizar.
	•	Distribución normalizada final.

Esto sirve como evidencia del correcto funcionamiento del motor de inferencia.

⸻

7. Cómo funciona la inferencia por enumeración (resumen para sustentación)

El método implementado es el clásico enumeration_ask (Russell & Norvig):
	1.	Se quiere calcular P(X \mid e), donde:
	•	X es la variable de consulta.
	•	e es la evidencia observada (por ejemplo, Rain=light, Maintenance=no).
	2.	El algoritmo recorre la red en orden topológico (padres antes que hijos).
	3.	Para cada valor posible x de la variable de consulta:
	•	Fija X = x en la evidencia.
	•	Llama recursivamente a _enumerate_all, que:
	•	Si la variable actual está en la evidencia: usa directamente su probabilidad.
	•	Si no está en la evidencia: suma sobre todos sus valores posibles.
	4.	Se obtiene una distribución no normalizada sobre los valores de X, que luego se normaliza para que sumen 1.

⸻

8. Cómo adaptarlo a otros dominios

El diseño es genérico: no está amarrado al ejemplo del tren.
Para usarlo en otro dominio (por ejemplo, diagnóstico médico, sistemas de recomendación, etc.):
	1.	Define la estructura de la red en un nuevo archivo de texto (por ejemplo estructura_mi_red.txt), con relaciones Padre -> Hijo.
	2.	Define las CPTs de cada nodo en otro archivo (por ejemplo cpts_mi_red.txt) siguiendo el mismo formato.
	3.	Ejecuta:

python proyecto3.py \
  --estructura estructura_mi_red.txt \
  --cpt cpts_mi_red.txt \
  --consulta MiVariable \
  --evidencia A=valA B=valB ...



Mientras:
	•	Los nombres de nodos sean consistentes entre estructura y CPTs.
	•	Las filas de las CPTs sumen 1.
	•	Los valores usados en evidencia existan en VALUES del nodo.

el motor seguirá funcionando sin necesidad de cambiar el código.

⸻

9. Resumen para entregar / sustentar

El proyecto cumple con:
	1.	Red Bayesiana
	•	Representación orientada a objetos (Node, BayesianNetwork).
	•	Lectura de estructura desde archivo.
	•	Lectura de tablas de probabilidad desde archivo.
	•	Función para mostrar estructura (print_structure).
	•	Función para visualizar CPTs en texto (print_cpts).
	2.	Motor de inferencia por enumeración
	•	Implementación del algoritmo enumeration_ask.
	•	Cálculo de P(X \mid evidencia) para cualquier variable X.
	•	Soporte de múltiples valores por variable (no solo booleanas).
	•	Generación de traza detallada usando --verbose.
	3.	Validación
	•	Implementado el ejemplo visto en clase (tren / cita) con:
	•	estructura_red.txt
	•	cpts_red.txt

⸻

