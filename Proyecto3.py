from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os


@dataclass
class Nodo:
    """
    Representa un nodo en una Red Bayesiana.
    
    Atributos:
        nombre: Identificador único del nodo
        valores: Lista de valores posibles que puede tomar el nodo
        padres: Lista de nombres de nodos padres
        cpt: Tabla de probabilidad condicional (Conditional Probability Table)
             Formato: {tupla_valores_padres: {valor_nodo: probabilidad}}
    """
    nombre: str
    valores: List[str]
    padres: List[str] = field(default_factory=list)
    cpt: Dict[Tuple[str, ...], Dict[str, float]] = field(default_factory=dict)

    def probabilidad(self, valor: str, asignacion_padres: Dict[str, str]) -> float:
        """
        Calcula P(self=valor | padres=asignacion_padres)
        
        Args:
            valor: Valor del nodo actual a consultar
            asignacion_padres: Diccionario con los valores de todos los padres
            
        Returns:
            Probabilidad condicional solicitada
            
        Raises:
            KeyError: Si no existe la entrada en la CPT para la combinación dada
        """
        # Construir la clave para la CPT basada en los valores de los padres
        clave = tuple(asignacion_padres[p] for p in self.padres) if self.padres else ()
        
        try:
            return self.cpt[clave][valor]
        except KeyError as e:
            raise KeyError(
                f"Falta entrada en CPT para nodo {self.nombre}, "
                f"padres {self.padres}, asignación {asignacion_padres}, valor {valor}"
            ) from e

    def __str__(self) -> str:
        """Representación en string del nodo para debugging"""
        padres_str = ", ".join(self.padres) if self.padres else "Ninguno"
        return f"Nodo({self.nombre}, valores={self.valores}, padres={padres_str})"


class RedBayesiana:
    """
    Implementa una Red Bayesiana con motor de inferencia por enumeración.
    
    La red se compone de nodos y relaciones padre-hijo, con tablas de probabilidad
    condicional para cada nodo.
    """
    
    def __init__(self):
        """Inicializa una red bayesiana vacía"""
        self.nodos: Dict[str, Nodo] = {}
        self.hijos: Dict[str, List[str]] = {}

    # ================== MANEJO DE ESTRUCTURA ==================

    def agregar_arista(self, padre: str, hijo: str):
        """
        Agrega una relación de dependencia padre -> hijo a la red.
        
        Args:
            padre: Nombre del nodo padre
            hijo: Nombre del nodo hijo
            
        Nota:
            Si los nodos no existen, se crean automáticamente
        """
        if padre not in self.nodos:
            self.nodos[padre] = Nodo(nombre=padre, valores=[])
        if hijo not in self.nodos:
            self.nodos[hijo] = Nodo(nombre=hijo, valores=[])

        # Actualizar lista de padres del hijo
        if padre not in self.nodos[hijo].padres:
            self.nodos[hijo].padres.append(padre)

        # Actualizar lista de hijos del padre
        self.hijos.setdefault(padre, [])
        if hijo not in self.hijos[padre]:
            self.hijos[padre].append(hijo)

    def definir_info_nodo(self, nombre: str, valores: List[str]):
        """
        Define el dominio (valores posibles) de un nodo.
        
        Args:
            nombre: Nombre del nodo
            valores: Lista de valores posibles del nodo
        """
        if nombre not in self.nodos:
            self.nodos[nombre] = Nodo(nombre=nombre, valores=valores)
        else:
            self.nodos[nombre].valores = valores

    def definir_entrada_cpt(self, nombre: str, valores_padres: Tuple[str, ...], 
                           probabilidades_valores: Dict[str, float]):
        """
        Define una fila de la CPT para un nodo.
        
        Args:
            nombre: Nombre del nodo
            valores_padres: Tupla con valores de los padres en orden
            probabilidades_valores: Diccionario valor -> probabilidad
            
        Raises:
            ValueError: Si las probabilidades no suman 1
        """
        nodo = self.nodos[nombre]
        total = sum(probabilidades_valores.values())
        
        # Validar que las probabilidades sumen 1
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"CPT para nodo {nombre} y padres {valores_padres} no suma 1 (suma={total:.6f})."
            )
        
        nodo.cpt[valores_padres] = probabilidades_valores

    def raices(self) -> List[str]:
        """Retorna la lista de nodos raíz (sin padres)"""
        return [nombre for nombre, nodo in self.nodos.items() if not nodo.padres]

    def orden_topologico(self) -> List[str]:
        """
        Calcula el orden topológico de los nodos (padres antes que hijos).
        
        Usa el algoritmo de Kahn para ordenamiento topológico.
        
        Returns:
            Lista de nombres de nodos en orden topológico
            
        Raises:
            ValueError: Si la red tiene ciclos
        """
        # Calcular grados de entrada
        grado_entrada = {nombre: len(nodo.padres) for nombre, nodo in self.nodos.items()}
        cola = [n for n, d in grado_entrada.items() if d == 0]
        orden: List[str] = []

        while cola:
            nodo_actual = cola.pop(0)
            orden.append(nodo_actual)
            
            # Reducir grado de entrada de hijos
            for hijo in self.hijos.get(nodo_actual, []):
                grado_entrada[hijo] -= 1
                if grado_entrada[hijo] == 0:
                    cola.append(hijo)

        # Verificar que todos los nodos fueron procesados
        if len(orden) != len(self.nodos):
            raise ValueError("La red tiene ciclos - no es un grafo acíclico dirigido.")
        
        return orden

    # ================== VALIDACIÓN ==================

    def validar_red(self):
        """
        Valida que la red esté completa y sea consistente.
        
        Verifica:
        - Todos los nodos tienen valores definidos
        - Las CPTs cubren todas las combinaciones de padres
        - Las probabilidades en cada fila de CPT suman 1
        
        Raises:
            ValueError: Si se encuentra algún problema en la red
        """
        for nombre, nodo in self.nodos.items():
            # Verificar que el nodo tenga valores definidos
            if not nodo.valores:
                raise ValueError(f"El nodo {nombre} no tiene valores definidos")
            
            # Para nodos con padres, verificar todas las combinaciones
            if nodo.padres:
                # Generar todas las combinaciones posibles de valores de padres
                combinaciones_padres = [[]]
                
                for padre_nombre in nodo.padres:
                    padre = self.nodos[padre_nombre]
                    nuevas_combinaciones = []
                    
                    for comb in combinaciones_padres:
                        for valor in padre.valores:
                            nuevas_combinaciones.append(comb + [valor])
                    combinaciones_padres = nuevas_combinaciones
                
                # Verificar que cada combinación tenga entrada en CPT
                for comb in combinaciones_padres:
                    clave = tuple(comb)
                    if clave not in nodo.cpt:
                        raise ValueError(
                            f"Falta entrada CPT para nodo {nombre} con padres {clave}"
                        )
            else:
                # Nodo sin padres debe tener al menos una entrada CPT
                if () not in nodo.cpt:
                    raise ValueError(f"Falta CPT para nodo raíz {nombre}")
            
            # Verificar que cada entrada CPT tenga probabilidades válidas
            for valores_padres, distribucion in nodo.cpt.items():
                total = sum(distribucion.values())
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"CPT para nodo {nombre} con padres {valores_padres} "
                        f"no suma 1 (suma={total:.6f})"
                    )

    # ================== VISUALIZACIÓN ==================

    def imprimir_estructura(self):
        """Imprime la estructura de la red de forma organizada"""
        print("\n" + "=" * 50)
        print("ESTRUCTURA DE LA RED BAYESIANA")
        print("=" * 50)
        
        raices = self.raices()
        print(f"Nodos raíz: {', '.join(raices) if raices else 'Ninguno'}")
        print()
        
        for nombre in self.orden_topologico():
            nodo = self.nodos[nombre]
            padres = ", ".join(nodo.padres) if nodo.padres else "Ninguno"
            hijos = ", ".join(self.hijos.get(nombre, [])) if self.hijos.get(nombre) else "Ninguno"
            
            print(f"NODO: {nombre}")
            print(f"  Padres: {padres}")
            print(f"  Hijos:  {hijos}")
            print(f"  Valores: {', '.join(nodo.valores)}")
            print()
        
        print("=" * 50)

    def imprimir_cpts(self):
        """Imprime las tablas de probabilidad condicional de forma organizada"""
        print("\n" + "=" * 50)
        print("TABLAS DE PROBABILIDAD CONDICIONAL (CPT)")
        print("=" * 50)
        
        for nombre in self.orden_topologico():
            nodo = self.nodos[nombre]
            print(f"\n--- Nodo: {nombre} ---")
            print(f"Valores: {', '.join(nodo.valores)}")
            
            if not nodo.padres:
                # Nodo sin padres
                distribucion = nodo.cpt.get((), {})
                print("Probabilidades:")
                for valor in nodo.valores:
                    prob = distribucion.get(valor, 0.0)
                    print(f"  P({nombre}={valor}) = {prob:.4f}")
            else:
                # Nodo con padres
                print(f"Padres: {', '.join(nodo.padres)}")
                print()
                
                # Encabezado de la tabla
                header_padres = " | ".join(nodo.padres)
                header_valores = " | ".join(nodo.valores)
                print(f"{header_padres} || {header_valores}")
                print("-" * (len(header_padres) + len(header_valores) + 4))
                
                # Filas de la tabla
                for asignacion_padres, distribucion in nodo.cpt.items():
                    str_padres = " | ".join(asignacion_padres)
                    str_valores = " | ".join(f"{distribucion.get(v, 0.0):.4f}" for v in nodo.valores)
                    print(f"{str_padres} || {str_valores}")
        
        print("\n" + "=" * 50)

    # ================== CARGA DESDE ARCHIVOS ==================

    @classmethod
    def desde_archivos(cls, ruta_estructura: str, ruta_cpt: str) -> "RedBayesiana":
        """
        Crea una red bayesiana desde archivos de texto.
        
        Args:
            ruta_estructura: Ruta al archivo con la estructura de la red
            ruta_cpt: Ruta al archivo con las tablas de probabilidad
            
        Returns:
            RedBayesiana cargada y validada
        """
        red = cls()

        # ----- CARGAR ESTRUCTURA -----
        print(f"\nCargando estructura desde: {ruta_estructura}")
        with open(ruta_estructura, "r", encoding="utf-8") as archivo:
            for num_linea, linea in enumerate(archivo, 1):
                linea = linea.strip()
                
                # Saltar líneas vacías o comentarios
                if not linea or linea.startswith("#"):
                    continue
                    
                if "->" not in linea:
                    raise ValueError(
                        f"Línea {num_linea} inválida: '{linea}'. "
                        f"Formato esperado: 'Padre -> Hijo'"
                    )
                
                # Procesar relación padre -> hijo
                padre_str, hijo_str = linea.split("->")
                padre = padre_str.strip()
                hijo = hijo_str.strip()
                red.agregar_arista(padre, hijo)

        # ----- CARGAR TABLAS DE PROBABILIDAD -----
        print(f"Cargando CPTs desde: {ruta_cpt}")
        with open(ruta_cpt, "r", encoding="utf-8") as archivo:
            lineas = [
                linea.strip()
                for linea in archivo
                if linea.strip() and not linea.strip().startswith("#")
            ]

        indice = 0
        while indice < len(lineas):
            linea = lineas[indice]
            
            # Inicio de definición de nodo
            if not linea.startswith("NODE"):
                raise ValueError(f"Se esperaba 'NODE <Nombre>', se encontró: {linea}")
            
            _, nombre_nodo = linea.split(None, 1)
            indice += 1

            # Leer valores del nodo
            partes = lineas[indice].split()
            if partes[0] != "VALUES":
                raise ValueError(
                    f"Se esperaba 'VALUES' después de NODE {nombre_nodo}, "
                    f"se encontró: {lineas[indice]}"
                )
            valores = partes[1:]
            red.definir_info_nodo(nombre_nodo, valores)
            indice += 1

            # Leer padres (opcional)
            padres: List[str] = []
            if indice < len(lineas) and lineas[indice].startswith("PARENTS"):
                padres = lineas[indice].split()[1:]
                
                # Actualizar relaciones padre-hijo
                for padre in padres:
                    if padre not in red.nodos:
                        red.nodos[padre] = Nodo(nombre=padre, valores=[])
                    if nombre_nodo not in red.nodos:
                        red.nodos[nombre_nodo] = Nodo(nombre=nombre_nodo, valores=valores)
                    
                    if padre not in red.nodos[nombre_nodo].padres:
                        red.nodos[nombre_nodo].padres.append(padre)
                    
                    red.hijos.setdefault(padre, [])
                    if nombre_nodo not in red.hijos[padre]:
                        red.hijos[padre].append(nombre_nodo)
                
                indice += 1

            # Leer tabla de probabilidades
            if indice >= len(lineas) or lineas[indice] != "TABLE":
                raise ValueError(
                    f"Se esperaba 'TABLE' para NODE {nombre_nodo}, "
                    f"se encontró: {lineas[indice] if indice < len(lineas) else 'EOF'}"
                )
            indice += 1

            # Procesar filas de la tabla
            while indice < len(lineas) and lineas[indice] != "ENDNODE":
                fila = lineas[indice].split()
                
                if padres:
                    # Nodo con padres: formato <valores_padres> <probabilidades>
                    if len(fila) != len(padres) + len(valores):
                        raise ValueError(
                            f"Línea de tabla mal formateada para nodo {nombre_nodo}: {lineas[indice]}"
                        )
                    valores_padres = tuple(fila[:len(padres)])
                    probabilidades = fila[len(padres):]
                else:
                    # Nodo sin padres: solo probabilidades
                    if len(fila) != len(valores):
                        raise ValueError(
                            f"Línea de tabla mal formateada para nodo {nombre_nodo} sin padres: {lineas[indice]}"
                        )
                    valores_padres = ()
                    probabilidades = fila

                # Crear diccionario de probabilidades
                dist_probabilidades = {val: float(prob) for val, prob in zip(valores, probabilidades)}
                red.definir_entrada_cpt(nombre_nodo, valores_padres, dist_probabilidades)
                indice += 1

            if indice >= len(lineas) or lineas[indice] != "ENDNODE":
                raise ValueError(f"Se esperaba 'ENDNODE' para NODE {nombre_nodo}")
            indice += 1

        # Validar la red completa
        print("Validando estructura y probabilidades...")
        red.validar_red()
        print("Red cargada y validada exitosamente!")
        
        return red

    # ================== INFERENCIA POR ENUMERACIÓN ==================

    def inferencia_por_enumeracion(self, variable_consulta: str, condiciones_observadas: Dict[str, str], 
                                 verbose: bool = False) -> Dict[str, float]:
        """
        Realiza inferencia por enumeración (algoritmo de Russell & Norvig).
        
        Args:
            variable_consulta: Variable para la cual calcular P(X|condiciones)
            condiciones_observadas: Diccionario con variables observadas y sus valores
            verbose: Si True, muestra traza detallada del cálculo
            
        Returns:
            Distribución de probabilidad P(X|condiciones)
        """
        if variable_consulta not in self.nodos:
            raise KeyError(f"Variable de consulta '{variable_consulta}' no existe en la red.")

        orden_variables = self.orden_topologico()
        nodo_consulta = self.nodos[variable_consulta]
        distribucion: Dict[str, float] = {}

        if verbose:
            print("\n" + "=" * 60)
            print(f"INICIANDO INFERENCIA PARA: {variable_consulta}")
            print(f"CONDICIONES OBSERVADAS: {self._formatear_condiciones(condiciones_observadas)}")
            print("=" * 60)

        # Calcular probabilidad para cada valor posible de la variable consulta
        for valor in nodo_consulta.valores:
            condiciones_extendidas = dict(condiciones_observadas)
            condiciones_extendidas[variable_consulta] = valor
            
            if verbose:
                print(f"\n--- Calculando P({variable_consulta}={valor} | {self._formatear_condiciones(condiciones_observadas)}) ---")
            
            probabilidad = self._enumerar_todas(orden_variables, condiciones_extendidas, verbose, 0)
            distribucion[valor] = probabilidad
            
            if verbose:
                print(f"Resultado sin normalizar: {probabilidad:.6f}")

        # Normalizar la distribución
        total = sum(distribucion.values())
        if total == 0:
            raise ValueError(
                "La probabilidad total es 0. "
                "Revise la red o las condiciones (pueden ser inconsistentes)."
            )
        
        for valor in distribucion:
            distribucion[valor] /= total

        return distribucion

    def _enumerar_todas(self, variables: List[str], condiciones: Dict[str, str], 
                       verbose: bool, profundidad: int) -> float:
        """
        Función recursiva auxiliar para el algoritmo de enumeración.
        
        Args:
            variables: Lista de variables pendientes por procesar (orden topológico)
            condiciones: Asignación actual de valores a variables
            verbose: Si True, muestra traza detallada
            profundidad: Nivel de recursión actual (para indentación)
            
        Returns:
            Probabilidad conjunta de las condiciones actuales
        """
        # Caso base: no hay más variables
        if not variables:
            return 1.0

        # Tomar primera variable
        variable_actual = variables[0]
        nodo_actual = self.nodos[variable_actual]
        variables_restantes = variables[1:]
        indentacion = "  " * profundidad

        # Obtener valores de los padres desde las condiciones
        asignacion_padres = {p: condiciones[p] for p in nodo_actual.padres if p in condiciones}

        if variable_actual in condiciones:
            # Variable en condiciones observadas: usar su valor fijo
            probabilidad = nodo_actual.probabilidad(condiciones[variable_actual], asignacion_padres)
            
            if verbose:
                print(f"{indentacion}{variable_actual} observada = {condiciones[variable_actual]}")
                print(f"{indentacion}P({variable_actual}={condiciones[variable_actual]} | padres) = {probabilidad:.6f}")
            
            return probabilidad * self._enumerar_todas(variables_restantes, condiciones, verbose, profundidad + 1)
        else:
            # Variable no observada: sumar sobre todos sus valores posibles
            total = 0.0
            
            if verbose:
                print(f"{indentacion}{variable_actual} no observada, sumando sobre: {nodo_actual.valores}")
            
            for valor_posible in nodo_actual.valores:
                probabilidad = nodo_actual.probabilidad(valor_posible, asignacion_padres)
                condiciones_extendidas = dict(condiciones)
                condiciones_extendidas[variable_actual] = valor_posible
                
                if verbose:
                    print(f"{indentacion}  Probando {variable_actual}={valor_posible}, P = {probabilidad:.6f}")
                
                # Llamada recursiva
                subtotal = probabilidad * self._enumerar_todas(variables_restantes, condiciones_extendidas, verbose, profundidad + 1)
                total += subtotal
                
                if verbose:
                    print(f"{indentacion}  Subtotal para {variable_actual}={valor_posible}: {subtotal:.6f}")
            
            if verbose:
                print(f"{indentacion}Total para {variable_actual}: {total:.6f}")
            
            return total

    def _formatear_condiciones(self, condiciones: Dict[str, str]) -> str:
        """
        Formatea las condiciones observadas para mostrarlas de manera legible.
        
        Args:
            condiciones: Diccionario de condiciones
            
        Returns:
            String formateado con las condiciones
        """
        if not condiciones:
            return "sin condiciones observadas"
        return ", ".join(f"{var}={valor}" for var, valor in condiciones.items())


def mostrar_menu():
    """Muestra el menú principal de la aplicación"""
    print("\n" + "=" * 60)
    print("MOTOR DE INFERENCIA POR ENUMERACIÓN - REDES BAYESIANAS")
    print("=" * 60)
    print("1. Cargar red bayesiana desde archivos")
    print("2. Mostrar estructura de la red")
    print("3. Mostrar tablas de probabilidad (CPT)")
    print("4. Realizar consulta de inferencia")
    print("5. Salir")
    print("-" * 60)


def obtener_archivos():
    """Solicita al usuario los archivos de estructura y CPT"""
    print("\n--- CARGA DE RED BAYESIANA ---")
    
    while True:
        archivo_estructura = input("Ingrese la ruta del archivo de estructura (ej: estructura_red.txt): ").strip()
        if not archivo_estructura:
            print("Debe ingresar un archivo de estructura.")
            continue
            
        if not os.path.exists(archivo_estructura):
            print(f"El archivo '{archivo_estructura}' no existe. Intente nuevamente.")
            continue
        break
    
    while True:
        archivo_cpt = input("Ingrese la ruta del archivo de CPT (ej: cpts_red.txt): ").strip()
        if not archivo_cpt:
            print("Debe ingresar un archivo de CPT.")
            continue
            
        if not os.path.exists(archivo_cpt):
            print(f"El archivo '{archivo_cpt}' no existe. Intente nuevamente.")
            continue
        break
    
    return archivo_estructura, archivo_cpt


def seleccionar_condiciones_interactivo(red, consulta_actual):
    """
    Permite al usuario seleccionar condiciones de forma interactiva.
    
    Args:
        red: La red bayesiana
        consulta_actual: Variable que se está consultando
        
    Returns:
        Diccionario con las condiciones seleccionadas
    """
    condiciones = {}
    nodos_disponibles = [n for n in red.orden_topologico() if n != consulta_actual]
    
    if not nodos_disponibles:
        print("No hay nodos disponibles para establecer condiciones.")
        return condiciones
    
    print(f"\n--- ESTABLECER CONDICIONES PARA P({consulta_actual} | condiciones) ---")
    print("Seleccione las variables que desea establecer como condiciones:")
    
    while True:
        # Mostrar nodos disponibles
        print("\nVariables disponibles para establecer condiciones:")
        for i, nombre in enumerate(nodos_disponibles, 1):
            nodo = red.nodos[nombre]
            print(f"  {i}. {nombre} ({', '.join(nodo.valores)})")
        print("  0. Terminar de establecer condiciones")
        
        try:
            seleccion = input("\nSeleccione una variable (número) o 0 para terminar: ").strip()
            if not seleccion:
                continue
                
            opcion = int(seleccion)
            if opcion == 0:
                break
                
            if opcion < 1 or opcion > len(nodos_disponibles):
                print(f"Selección inválida. Elija un número entre 1 y {len(nodos_disponibles)}.")
                continue
                
            variable_seleccionada = nodos_disponibles[opcion - 1]
            nodo = red.nodos[variable_seleccionada]
            
            # Seleccionar valor para la variable
            print(f"\nSeleccione el valor para {variable_seleccionada}:")
            for i, valor in enumerate(nodo.valores, 1):
                print(f"  {i}. {valor}")
                
            while True:
                try:
                    seleccion_valor = input(f"Seleccione el valor para {variable_seleccionada} (1-{len(nodo.valores)}): ").strip()
                    if not seleccion_valor:
                        continue
                        
                    opcion_valor = int(seleccion_valor)
                    if opcion_valor < 1 or opcion_valor > len(nodo.valores):
                        print(f"Selección inválida. Elija un número entre 1 y {len(nodo.valores)}.")
                        continue
                        
                    valor_seleccionado = nodo.valores[opcion_valor - 1]
                    condiciones[variable_seleccionada] = valor_seleccionado
                    print(f"Condición establecida: {variable_seleccionada} = {valor_seleccionado}")
                    
                    # Remover la variable de las disponibles
                    nodos_disponibles.remove(variable_seleccionada)
                    break
                    
                except ValueError:
                    print("Por favor ingrese un número válido.")
                    
        except ValueError:
            print("Por favor ingrese un número válido.")
            
    return condiciones


def realizar_consulta(red):
    """Realiza una consulta de inferencia interactiva"""
    if not red.nodos:
        print("Primero debe cargar una red bayesiana (opción 1).")
        return
    
    print("\n--- REALIZAR CONSULTA DE INFERENCIA ---")
    
    # Mostrar nodos disponibles
    print("Nodos disponibles en la red:")
    for i, nombre in enumerate(red.orden_topologico(), 1):
        nodo = red.nodos[nombre]
        print(f"  {i}. {nombre} ({', '.join(nodo.valores)})")
    
    # Seleccionar variable de consulta
    while True:
        consulta = input("\nIngrese el nombre del nodo a consultar: ").strip()
        if consulta in red.nodos:
            break
        print(f"El nodo '{consulta}' no existe en la red. Intente nuevamente.")
    
    # Establecer condiciones de forma interactiva
    condiciones = seleccionar_condiciones_interactivo(red, consulta)
    
    # Preguntar por modo verbose
    print(f"\n¿Mostrar traza detallada del cálculo de P({consulta} | condiciones)?")
    verbose = input("(s/n, presione Enter para 'no'): ").strip().lower() == 's'
    
    try:
        # Realizar inferencia
        print(f"\nCalculando P({consulta} | {red._formatear_condiciones(condiciones)})...")
        
        distribucion = red.inferencia_por_enumeracion(consulta, condiciones, verbose=verbose)
        
        # Mostrar resultados
        print("\n" + "=" * 60)
        print("RESULTADO DE LA INFERENCIA")
        print("=" * 60)
        print(f"Distribución de probabilidad para {consulta} dado:")
        print(f"  Condiciones: {red._formatear_condiciones(condiciones)}")
        print("-" * 60)
        
        for valor, probabilidad in distribucion.items():
            print(f"  P({consulta}={valor} | condiciones) = {probabilidad:.6f}")
        
        # Mostrar el valor más probable
        valor_max = max(distribucion, key=distribucion.get)
        prob_max = distribucion[valor_max]
        print("-" * 60)
        print(f"Valor más probable: {consulta}={valor_max} con probabilidad {prob_max:.6f}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error durante la inferencia: {e}")


def main():
    """Función principal del programa con interfaz interactiva"""
    red = None
    
    print("Bienvenido al Motor de Inferencia por Enumeración")
    print("Este programa permite realizar inferencias probabilísticas usando Redes Bayesianas")
    
    while True:
        mostrar_menu()
        opcion = input("\nSeleccione una opción (1-5): ").strip()
        
        if opcion == "1":
            try:
                arch_estructura, arch_cpt = obtener_archivos()
                red = RedBayesiana.desde_archivos(arch_estructura, arch_cpt)
                print("\nRed cargada exitosamente!")
                red.imprimir_estructura()
                red.imprimir_cpts()
            except Exception as e:
                print(f"Error al cargar la red: {e}")
                red = None
        
        elif opcion == "2":
            if red:
                red.imprimir_estructura()
            else:
                print("Primero debe cargar una red bayesiana (opción 1).")
        
        elif opcion == "3":
            if red:
                red.imprimir_cpts()
            else:
                print("Primero debe cargar una red bayesiana (opción 1).")
        
        elif opcion == "4":
            if red:
                realizar_consulta(red)
            else:
                print("Primero debe cargar una red bayesiana (opción 1).")
        
        elif opcion == "5":
            print("\nGracias por usar el Motor de Inferencia por Enumeración. ¡Hasta pronto!")
            break
        
        else:
            print("Opción no válida. Por favor, seleccione una opción del 1 al 5.")
        
        input("\nPresione Enter para continuar...")


if __name__ == "__main__":
    main()