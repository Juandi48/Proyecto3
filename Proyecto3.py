from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import argparse


@dataclass
class Node:
    """
    Nodo de una Red Bayesiana.
    - name: nombre de la variable (cadena)
    - values: lista de valores posibles (dominio discreto)
    - parents: lista de nombres de variables padre
    - cpt: tabla de probabilidad condicional
           dict: (valores_padres_en_orden) -> { valor_propio: prob }
    """
    name: str
    values: List[str]
    parents: List[str] = field(default_factory=list)
    cpt: Dict[Tuple[str, ...], Dict[str, float]] = field(default_factory=dict)

    def prob(self, value: str, parent_assignment: Dict[str, str]) -> float:
        """
        Retorna P(self=value | padres=parent_assignment).
        parent_assignment contiene asignaciones para todos los padres.
        """
        key = tuple(parent_assignment[p] for p in self.parents) if self.parents else ()
        try:
            return self.cpt[key][value]
        except KeyError as e:
            raise KeyError(
                f"Falta entrada en CPT para nodo {self.name}, "
                f"padres {self.parents}, asignación {parent_assignment}, valor {value}"
            ) from e

    def __str__(self) -> str:
        parent_str = ", ".join(self.parents) if self.parents else "None"
        return f"Node({self.name}, values={self.values}, parents={parent_str})"


class BayesianNetwork:
    """
    Implementación básica de Red Bayesiana + motor de inferencia por enumeración.
    """
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.children: Dict[str, List[str]] = {}

    # ================== Manejo de estructura ==================

    def add_edge(self, parent: str, child: str):
        """
        Agregar arista padre -> hijo a la red.
        Se crean los nodos si no existían.
        """
        if parent not in self.nodes:
            self.nodes[parent] = Node(name=parent, values=[])
        if child not in self.nodes:
            self.nodes[child] = Node(name=child, values=[])

        if child not in self.nodes[child].parents:
            self.nodes[child].parents.append(parent)

        self.children.setdefault(parent, [])
        if child not in self.children[parent]:
            self.children[parent].append(child)

    def set_node_info(self, name: str, values: List[str]):
        """
        Definir el dominio (valores posibles) de un nodo.
        """
        if name not in self.nodes:
            self.nodes[name] = Node(name=name, values=values)
        else:
            self.nodes[name].values = values

    def set_cpt_entry(self, name: str, parent_values: Tuple[str, ...], value_probs: Dict[str, float]):
        """
        Definir una fila de la CPT para el nodo `name`.
        parent_values es una tupla con los valores de los padres en orden.
        value_probs es un dict valor_propio -> prob.
        """
        node = self.nodes[name]
        total = sum(value_probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"CPT para nodo {name} y padres {parent_values} no suma 1 (suma={total})."
            )
        node.cpt[parent_values] = value_probs

    def roots(self) -> List[str]:
        """Retorna la lista de nodos raíz (sin padres)."""
        return [name for name, node in self.nodes.items() if not node.parents]

    def topological_order(self) -> List[str]:
        """
        Orden topológico (padres antes que hijos).
        Implementación de algoritmo de Kahn.
        """
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}
        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []

        while queue:
            n = queue.pop(0)
            order.append(n)
            for child in self.children.get(n, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("La red tiene un ciclo o algo raro: no se pudo ordenar topológicamente.")
        return order

    # ================== Impresión / visualización ==================

    def print_structure(self):
        """
        Imprime en texto la estructura de la red:
        para cada nodo, sus padres e hijos, recorridos en orden topológico.
        """
        print("=== Estructura de la Red Bayesiana ===")
        roots = self.roots()
        print(f"Nodos raíz: {', '.join(roots) if roots else 'Ninguno'}")
        for name in self.topological_order():
            node = self.nodes[name]
            parents = ", ".join(node.parents) if node.parents else "None"
            childs = ", ".join(self.children.get(name, [])) if self.children.get(name) else "None"
            print(f"- {name}")
            print(f"    Padres: {parents}")
            print(f"    Hijos : {childs}")
        print("=== Fin de estructura ===\n")

    def print_cpts(self):
        """
        Imprime en texto las tablas de probabilidad condicional (CPT).
        """
        print("=== Tablas de Probabilidad Condicional (CPT) ===")
        for name in self.topological_order():
            node = self.nodes[name]
            print(f"Nodo: {name}")
            print(f"Valores: {', '.join(node.values)}")
            if not node.parents:
                dist = node.cpt.get((), {})
                for val in node.values:
                    print(f"  P({name}={val}) = {dist.get(val, 'N/A')}")
            else:
                print(f"Padres: {', '.join(node.parents)}")
                header = "  " + "  ".join(node.parents) + "  |  " + "  ".join(node.values)
                print(header)
                for parent_assign, dist in node.cpt.items():
                    parent_vals_str = "  ".join(parent_assign)
                    probs_str = "  ".join(f"{dist.get(v, 'N/A')}" for v in node.values)
                    print(f"  {parent_vals_str}  |  {probs_str}")
            print()
        print("=== Fin de tablas CPT ===\n")

    # ================== Carga desde archivos ==================

    @classmethod
    def from_files(cls, structure_path: str, cpt_path: str) -> "BayesianNetwork":
        """
        Carga la red desde dos archivos de texto:
        - structure_path: cada línea 'Padre -> Hijo'
        - cpt_path: formato por nodos (ver abajo).
        """
        bn = cls()

        # ----- Estructura -----
        with open(structure_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "->" not in line:
                    raise ValueError(
                        f"Línea de estructura inválida (se esperaba 'Padre -> Hijo'): {line}"
                    )
                parent_str, child_str = line.split("->")
                parent = parent_str.strip()
                child = child_str.strip()
                bn.add_edge(parent, child)

        # ----- CPTs -----
        with open(cpt_path, "r", encoding="utf-8") as f:
            content = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        i = 0
        while i < len(content):
            line = content[i]
            if not line.startswith("NODE"):
                raise ValueError(f"Se esperaba 'NODE <Nombre>' y se encontró: {line}")
            _, node_name = line.split(None, 1)
            i += 1

            # VALUES
            parts = content[i].split()
            if parts[0] != "VALUES":
                raise ValueError(
                    f"Se esperaba 'VALUES' después de NODE {node_name}, "
                    f"se encontró: {content[i]}"
                )
            values = parts[1:]
            bn.set_node_info(node_name, values)
            i += 1

            # PARENTS (opcional)
            parents: List[str] = []
            if i < len(content) and content[i].startswith("PARENTS"):
                parents = content[i].split()[1:]
                # Ajustar padres en el nodo y estructura
                for p in parents:
                    if p not in bn.nodes:
                        bn.nodes[p] = Node(name=p, values=[])
                    if node_name not in bn.nodes:
                        bn.nodes[node_name] = Node(name=node_name, values=values)
                    if p not in bn.nodes[node_name].parents:
                        bn.nodes[node_name].parents.append(p)
                    bn.children.setdefault(p, [])
                    if node_name not in bn.children[p]:
                        bn.children[p].append(node_name)
                i += 1

            # TABLE
            if i >= len(content) or content[i] != "TABLE":
                raise ValueError(
                    f"Se esperaba 'TABLE' para NODE {node_name}, "
                    f"se encontró: {content[i] if i < len(content) else 'EOF'}"
                )
            i += 1

            # Filas de la tabla hasta ENDNODE
            while i < len(content) and content[i] != "ENDNODE":
                row = content[i].split()
                if parents:
                    # formato: <padres...> <p(valor1)> <p(valor2)> ...
                    if len(row) != len(parents) + len(values):
                        raise ValueError(
                            f"Línea de tabla mal formateada para nodo {node_name}: {content[i]}"
                        )
                    parent_vals = tuple(row[:len(parents)])
                    prob_vals = row[len(parents):]
                else:
                    # sin padres: solo las probabilidades
                    if len(row) != len(values):
                        raise ValueError(
                            f"Línea de tabla mal formateada para nodo {node_name} sin padres: {content[i]}"
                        )
                    parent_vals = ()
                    prob_vals = row

                value_probs = {val: float(p) for val, p in zip(values, prob_vals)}
                bn.set_cpt_entry(node_name, parent_vals, value_probs)
                i += 1

            if i >= len(content) or content[i] != "ENDNODE":
                raise ValueError(f"Se esperaba 'ENDNODE' para NODE {node_name}")
            i += 1

        return bn

    # ================== Inferencia por enumeración ==================

    def enumeration_ask(self, query_var: str, evidence: Dict[str, str], verbose: bool = False) -> Dict[str, float]:
        """
        Implementa el algoritmo de inferencia por enumeración (Russell & Norvig).
        Retorna la distribución P(query_var | evidence).
        """
        if query_var not in self.nodes:
            raise KeyError(f"Variable de consulta '{query_var}' no existe en la red.")

        vars_order = self.topological_order()
        node = self.nodes[query_var]
        dist: Dict[str, float] = {}

        for value in node.values:
            extended_evidence = dict(evidence)
            extended_evidence[query_var] = value
            if verbose:
                print(f"--- Calculando término para {query_var}={value} dado evidencia {evidence} ---")
            dist[value] = self._enumerate_all(vars_order, extended_evidence, verbose=verbose, depth=0)
            if verbose:
                print(f"Resultado sin normalizar para {query_var}={value}: {dist[value]}\n")

        # Normalizar
        total = sum(dist.values())
        if total == 0:
            raise ValueError("La probabilidad total es 0; revise la red o la evidencia.")
        for v in dist:
            dist[v] /= total

        if verbose:
            print(f"Distribución normalizada para {query_var} dado {evidence}: {dist}\n")

        return dist

    def _enumerate_all(self, vars_order: List[str], evidence: Dict[str, str], verbose: bool, depth: int) -> float:
        """
        Parte recursiva del algoritmo de enumeración.
        """
        if not vars_order:
            return 1.0

        Y = vars_order[0]
        nodeY = self.nodes[Y]
        rest = vars_order[1:]
        indent = "  " * depth

        # Asignación de padres para Y a partir de la evidencia
        parent_assignment = {p: evidence[p] for p in nodeY.parents if p in evidence}

        if Y in evidence:
            probY = nodeY.prob(evidence[Y], parent_assignment)
            if verbose:
                print(f"{indent}Y={Y} está en la evidencia como {evidence[Y]}, P={probY}")
            return probY * self._enumerate_all(rest, evidence, verbose, depth + 1)
        else:
            total = 0.0
            if verbose:
                print(f"{indent}Y={Y} no está en la evidencia, sumando sobre sus valores...")
            for y_val in nodeY.values:
                probY = nodeY.prob(y_val, parent_assignment)
                if verbose:
                    print(f"{indent}  Asignando {Y}={y_val}, P={probY}")
                evidence_extended = dict(evidence)
                evidence_extended[Y] = y_val
                subtotal = probY * self._enumerate_all(rest, evidence_extended, verbose, depth + 1)
                if verbose:
                    print(f"{indent}  Subtotal para {Y}={y_val}: {subtotal}")
                total += subtotal
            if verbose:
                print(f"{indent}Total para Y={Y}: {total}")
            return total


def main():
    parser = argparse.ArgumentParser(
        description="Motor de inferencia por enumeración con Redes Bayesianas."
    )
    parser.add_argument(
        "--estructura", "-s", required=True,
        help="Ruta al archivo de estructura de la red (líneas 'Padre -> Hijo')."
    )
    parser.add_argument(
        "--cpt", "-c", required=True,
        help="Ruta al archivo con las tablas de probabilidad (CPT)."
    )
    parser.add_argument(
        "--consulta", "-q", required=False,
        help="Variable de consulta (nombre de nodo en la red)."
    )
    parser.add_argument(
        "--evidencia", "-e", nargs="*",
        help="Evidencia en formato Var=valor, por ejemplo Rain=light"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Mostrar traza detallada (paso a paso) de la inferencia."
    )
    args = parser.parse_args()

    # Cargar la red
    bn = BayesianNetwork.from_files(args.estructura, args.cpt)

    # Mostrar estructura y CPTs
    bn.print_structure()
    bn.print_cpts()

    # Si hay consulta, hacer inferencia
    if args.consulta:
        evidence_dict: Dict[str, str] = {}
        if args.evidencia:
            for item in args.evidencia:
                if "=" not in item:
                    raise ValueError(
                        f"Formato de evidencia inválido: {item}. Debe ser Var=valor"
                    )
                var, val = item.split("=", 1)
                evidence_dict[var] = val
        print(f"Realizando inferencia para {args.consulta} dado evidencia {evidence_dict}")
        dist = bn.enumeration_ask(args.consulta, evidence_dict, verbose=args.verbose)
        print(f"Distribución de probabilidad de {args.consulta} dado la evidencia:")
        for val, p in dist.items():
            print(f"  P({args.consulta}={val} | evidencia) = {p:.5f}")
    else:
        print("No se especificó variable de consulta. Solo se cargó y mostró la red y las CPT.")


if __name__ == "__main__":
    main()