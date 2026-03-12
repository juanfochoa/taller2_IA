from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    
    assignment = {}
    return backtrack(csp, assignment)
  
def backtrack(csp: DroneAssignmentCSP, assignment: dict[str, str]):
  if csp.is_complete(assignment):
    return assignment
  
  unassigned_vars = csp.get_unassigned_variables(assignment)
  var = unassigned_vars[0]
  
  for value in csp.domains[var]:
    if csp.is_consistent(var, value, assignment):
      csp.assign(var, value, assignment)
      
      result = backtrack(csp, assignment)
      if result is not None:
        return result
      csp.unassign(var, assignment)
  
  return None


def backtracking_fc(csp: DroneAssignmentCSP, assignment: dict[str, str]) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    if assignment is None:
      assignment = {}
      
    if csp.is_complete(assignment):
      return assignment
    
    unassigned_vars = csp.get_unassigned_variables(assignment)
    var = unassigned_vars[0]
    
    for value in csp.domains[var]:
      if csp.is_consistent(var, value, assignment):
        csp.assign(var, value, assignment)
        
        domains_saved = {}
        for d in csp.domains:
          domains_saved[d] = list(csp.domains[d])
          
        fail = False
        
        for neighbor in csp.get_neighbors(var):
          if neighbor not in assignment:
            new_domain = []
            
            for val in csp.domains[neighbor]:
              #se verifica que el valor que sigue siendo consistente
              if csp.is_consistent(neighbor, val, assignment):
                #si es valido, se mantiene
                new_domain.append(val)
            
            #Asigno ese nuevo dominio con las nuevas asignaciones, o las restricciones aplicadas
            csp.domains[neighbor] = new_domain    
            
            if len(new_domain) == 0:
              fail = True
              break
        
        #Si ningun dominio quedo vacio sigo explorando
        if not fail:
          result = backtracking_fc(csp, assignment)
          if result is not None:
            return result

        csp.domains = domains_saved
        csp.unassign(var, assignment)
  
    return None
        


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    # crea la asignación base
    assignment = {}

    if not ac3(csp):
        return None

    return backtrack(csp, assignment)

def revise(csp, Xi, Xj):
    #Para cada valor de Xi debe existir por lo menos un valor compatible en Xj
    
    revised = False
    #se revisan los valores
    for x in csp.domains[Xi][:]:
      supported = False
      
      for y in csp.domains[Xj]:
        assignment = {Xi: x, Xj: y}
        if csp.is_consistent(Xi, x, assignment):
          supported = True
          break
        #si no hay soporte en Xj se elimina
        if not supported:
          csp.domains[Xi].remove(x)
          revised = True
      return revised
    
def ac3(csp):
    #crea la cola de arcos
    queue = []
    
    for var in csp.domains:
      for neighbor in csp.get_neighbors(var):
        queue.append((var, neighbor))
    
    while queue:
      Xi, Xj = queue.pop(0)
      #se intenta eliminar valores invalidos de Xi
      if revise(csp, Xi, Xj):
        #Se verifica que el dominio este vacio
        if len(csp.domains[Xi]) == 0:
          return False
        #agrega nuevos arcos, propagando las restricciones a otros vecinos
        for Xk in csp.get_neighbors(Xi):
          if Xk != Xj:
            queue.append((Xk, Xi))
    return True
  
def backtrack(csp, assignment):
    #verifica si el problema esta completo
    if csp.is_complete(assignment):
      return assignment
    
    #escoge una variable sin asignar
    var = csp.get_unassigned_variables(assignment)[0]
    
    # prueba con cada valor del dominio
    for value in csp.domains[var]:
      if csp.is_consistent(var, value, assignment):
        csp.assign(var, value, assignment)
        #guarda los dominios para poder restaurarlos en caso de backtracking
        saved_domains = {v: list(csp.domains[v]) for v in csp.domains}
        
        #intenta reducir los dominios despues de la asignación
        if ac3(csp):
          #sigue buscando de manera recursiva
          result = backtrack(csp, assignment)
          if result is not None:
            return result
        # si no funciono restaura los dominios
        csp.domains = saved_domains
        # y se deshace de la asignación
        csp.unassign(var, assignment)

    return None
    

def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    #(BONUS)
    def backtrack(assigment):
      if csp.is_complete(assigment):
        return assigment
      
      #MRV: se escoge la variable con menos valores legales
      unassigned_vars = csp.get_unassigned_variables(assigment)

      var = min(unassigned_vars, key=lambda v: len(csp.domains[v]))

      #LCV: se ordenan los valores por el que menos restricciones impone a los vecinos
      values = sorted(
        csp.domains[var], 
        key=lambda val: csp.get_num_conflicts(var, val, assigment)
      )

      for value in values:
        if csp.is_consistent(var,value, assigment):
          csp.assign(var, value, assigment)
          
          # guardar dominios para restaurar en caso de backtracking
          saved_domains = {v: list(csp.domains[v]) for v in csp.domains}

          #forward checking
          fail = False
          for neighbor in csp.get_neighbors(var):
            if neighbor not in assigment:
              new_domain = []
              for val in csp.domains[neighbor]:
                if csp.is_consistent(neighbor, val, assigment):
                  new_domain.append(val)
              csp.domains[neighbor] = new_domain
              if len(new_domain) == 0:
                fail = True
                break
          
          if not fail:
            result = backtrack(assigment)
            if result is not None:
              return result
          
          #restaurar dominios
          csp.domains = saved_domains
          csp.unassign(var, assigment)
      return None
    return backtrack({})
