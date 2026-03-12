from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.utils import bfs_distance


if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    # TODO: Implement your code here
    """if state.is_win():
        return 1000

    if state.is_lose():
        return -1000

    drone_pos = state.get_drone_position()
    hunters = state.get_hunter_positions()
    deliveries = state.get_pending_deliveries()
    layout = state.get_layout()

    score = state.get_score()

    # distancia a la entrega más cercana
    if deliveries:
        dist_delivery = min(
            bfs_distance(layout, drone_pos, d, False) for d in deliveries
        )
    else:
        dist_delivery = 0

    # distancia al hunter más cercano
    if hunters:
        dist_hunter = min(
            bfs_distance(layout, h, drone_pos, True) for h in hunters
        )
    else:
        dist_hunter = float("inf")

    value = 0

    # preferir estar cerca de entregas
    value -= dist_delivery * 5

    # preferir estar lejos de hunters
    if dist_hunter != float("inf"):
        value += dist_hunter * 3

    # menos entregas pendientes es mejor
    value -= len(deliveries) * 20

    # sumar score actual
    value += score

    return max(-1000, min(1000, value))"""
    
    #Mejoras hechas con ayuda de la IA para hacer la función de evaluación más sofisticada y efectiva:
    """ PROMT: Mejora la función de evaluación para el juego del dron vs. cazadores, considerando no solo la 
    distancia a la entrega más cercana y a los cazadores, sino también la urgencia de las entregas 
    (si el dron puede llegar antes que los cazadores), el peligro de estar cerca de cazadores, y una 
    bonificación por entregas seguras. Ajusta los pesos para priorizar avanzar hacia entregas seguras 
    y huir de cazadores cercanos. Asegúrate de manejar casos sin entregas o sin cazadores adecuadamente. """

    if state.is_win():
        return 1000
    if state.is_lose():
        return -1000

    drone_pos = state.get_drone_position()
    hunters = state.get_hunter_positions()
    deliveries = state.get_pending_deliveries()
    layout = state.get_layout()
    score = state.get_score()

    # CAMBIO 1: Se calcula la distancia BFS a cada entrega (no solo la mínima todavía)
    # para poder comparar si el dron llega antes que cualquier cazador (urgencia de entrega)
    delivery_list = list(deliveries)

    if delivery_list:
        drone_to_deliveries = [
            bfs_distance(layout, drone_pos, d, False) for d in delivery_list
        ]
        dist_delivery = min(drone_to_deliveries)
    else:
        dist_delivery = 0
        drone_to_deliveries = []

    # CAMBIO 2: Se calcula la distancia de cada cazador al dron con hunter_restricted=True
    # (igual que antes), pero ahora también se calcula la distancia de cada cazador
    # a cada punto de entrega, para detectar si el cazador "bloquea" una entrega
    if hunters:
        hunter_to_drone = [
            bfs_distance(layout, h, drone_pos, True) for h in hunters
        ]
        dist_hunter = min(hunter_to_drone)

        # CAMBIO 3: Peligro por entrega bloqueada — si un cazador está más cerca
        # de una entrega que el dron, esa entrega es "peligrosa"; se penaliza menos
        # ir hacia entregas donde el dron llega primero (urgencia segura)
        safe_delivery_bonus = 0
        for i, d_dist in enumerate(drone_to_deliveries):
            hunter_to_delivery = min(
                bfs_distance(layout, h, delivery_list[i], True) for h in hunters
            )
            if d_dist < hunter_to_delivery:
                # El dron llega antes: recompensar ir hacia esa entrega
                safe_delivery_bonus += 30

    else:
        dist_hunter = float("inf")
        safe_delivery_bonus = len(delivery_list) * 30  # sin cazadores, todas son seguras

    value = 0

    # CAMBIO 4: Peso mayor a la distancia de entrega (era 5, ahora 10)
    # porque queremos que el dron priorice avanzar hacia entregas
    value -= dist_delivery * 10

    # CAMBIO 5: Penalización más agresiva cuando el cazador está muy cerca
    # En lugar de recompensar distancia lineal, se usa una función inversa
    # para que el peligro crezca exponencialmente al acercarse
    if dist_hunter != float("inf"):
        if dist_hunter <= 2:
            # CAMBIO 6: Peligro crítico si el cazador está a 1-2 pasos: huir es prioridad
            value -= 500
        elif dist_hunter <= 4:
            value += dist_hunter * 20
        else:
            value += dist_hunter * 8

    # CAMBIO 7: Bonificación por entregas seguras (dron llega antes que cazadores)
    value += safe_delivery_bonus

    # Sin cambio: penalización por entregas pendientes
    value -= len(deliveries) * 20

    # Sin cambio: score actual
    value += score

    return max(-1000, min(1000, value))
