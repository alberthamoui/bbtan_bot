"""
physics.py — Motor de simulação de trajetória da bola.

Simula o caminho de uma bola disparada em um dado ângulo,
calculando colisões com paredes, blocos (AABB) e triângulos,
retornando uma pontuação para cada ângulo testado.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import config as cfg
from vision import Block, BlockType, PowerUp, PowerUpType, TriangleOrientation

# ─────────────────────────────────────────────
# RESULTADO DA SIMULAÇÃO
# ─────────────────────────────────────────────


@dataclass
class SimResult:
    """Resultado de uma simulação de trajetória."""

    angle: float  # ângulo testado em graus
    score: float  # pontuação total
    total_hits: int  # total de colisões com blocos
    blocks_destroyed: int  # blocos que seriam destruídos
    powerups_collected: int  # power-ups coletados (todos os tipos)
    extra_balls_collected: int = 0  # apenas power-ups do tipo EXTRA_BALL (+)
    path: List[Tuple[float, float]] = field(default_factory=list)
    hit_details: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# VETOR 2D SIMPLES (evita dependência de pygame)
# ─────────────────────────────────────────────


class Vec2:
    """Vetor 2D leve para física."""

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalized(self) -> "Vec2":
        l = self.length()
        if l < 1e-8:
            return Vec2(0, 0)
        return Vec2(self.x / l, self.y / l)

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y

    def reflect(self, normal: "Vec2") -> "Vec2":
        """Reflete este vetor em relação à normal dada."""
        d = self.dot(normal)
        return Vec2(self.x - 2 * d * normal.x, self.y - 2 * d * normal.y)

    def copy(self) -> "Vec2":
        return Vec2(self.x, self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


# ─────────────────────────────────────────────
# COLISÃO AABB (blocos quadrados)
# ─────────────────────────────────────────────


def _collide_aabb(
    pos: Vec2, vel: Vec2, radius: float, bx: int, by: int, bw: int, bh: int
) -> Optional[Vec2]:
    """
    Verifica colisão círculo vs AABB.
    Retorna a normal de reflexão ou None se não colidir.
    """
    # Ponto mais próximo do centro da bola no retângulo
    closest_x = max(bx, min(pos.x, bx + bw))
    closest_y = max(by, min(pos.y, by + bh))

    dx = pos.x - closest_x
    dy = pos.y - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return None  # sem colisão

    # Determina qual face foi atingida baseado na penetração
    # Calcula overlap em cada eixo
    overlap_left = (pos.x + radius) - bx
    overlap_right = (bx + bw) - (pos.x - radius)
    overlap_top = (pos.y + radius) - by
    overlap_bottom = (by + bh) - (pos.y - radius)

    min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

    if min_overlap == overlap_left and vel.x > 0:
        return Vec2(-1, 0)  # bateu na face esquerda
    elif min_overlap == overlap_right and vel.x < 0:
        return Vec2(1, 0)  # bateu na face direita
    elif min_overlap == overlap_top and vel.y > 0:
        return Vec2(0, -1)  # bateu no topo
    elif min_overlap == overlap_bottom and vel.y < 0:
        return Vec2(0, 1)  # bateu na base
    else:
        # Fallback: normal baseada na direção do centro
        if dist_sq < 1e-8:
            return Vec2(0, -1)
        inv_dist = 1.0 / math.sqrt(dist_sq)
        return Vec2(dx * inv_dist, dy * inv_dist)


# ─────────────────────────────────────────────
# COLISÃO AABBx2 (blocos quadrados grandes)
# ─────────────────────────────────────────────

def _collide_aabb2(
    pos: Vec2, vel: Vec2, radius: float, bx: int, by: int, bw: int, bh: int
) -> Optional[Vec2]:
    """
    Verifica colisão círculo vs AABB.
    Retorna a normal de reflexão ou None se não colidir.
    """
    # Ponto mais próximo do centro da bola no retângulo
    closest_x = max(bx, min(pos.x, bx + bw))
    closest_y = max(by, min(pos.y, by + bh))

    dx = pos.x - closest_x
    dy = pos.y - closest_y
    dist_sq = dx * dx + dy * dy

    if dist_sq >= radius * radius:
        return None  # sem colisão

    # Determina qual face foi atingida baseado na penetração
    # Calcula overlap em cada eixo
    overlap_left = (pos.x + radius) - bx
    overlap_right = (bx + bw) - (pos.x - radius)
    overlap_top = (pos.y + radius) - by
    overlap_bottom = (by + bh) - (pos.y - radius)

    min_overlap = min(overlap_left, overlap_right, overlap_top, overlap_bottom)

    if min_overlap == overlap_left and vel.x > 0:
        return Vec2(-1, 0)  # bateu na face esquerda
    elif min_overlap == overlap_right and vel.x < 0:
        return Vec2(1, 0)  # bateu na face direita
    elif min_overlap == overlap_top and vel.y > 0:
        return Vec2(0, -1)  # bateu no topo
    elif min_overlap == overlap_bottom and vel.y < 0:
        return Vec2(0, 1)  # bateu na base
    else:
        # Fallback: normal baseada na direção do centro
        if dist_sq < 1e-8:
            return Vec2(0, -1)
        inv_dist = 1.0 / math.sqrt(dist_sq)
        return Vec2(dx * inv_dist, dy * inv_dist)


# ─────────────────────────────────────────────
# COLISÃO COM TRIÂNGULO
# ─────────────────────────────────────────────


def _get_triangle_diagonal_normal(orientation: TriangleOrientation) -> Vec2:
    """
    Retorna a normal da hipotenusa do triângulo baseada na orientação.
    A normal aponta para FORA da superfície diagonal.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    normals = {
        # Triângulo com vértice no top-left: diagonal vai de top-right a bottom-left
        TriangleOrientation.TOP_LEFT: Vec2(-inv_sqrt2, -inv_sqrt2),
        # Triângulo com vértice no top-right: diagonal vai de top-left a bottom-right
        TriangleOrientation.TOP_RIGHT: Vec2(inv_sqrt2, -inv_sqrt2),
        # Triângulo com vértice no bottom-left
        TriangleOrientation.BOTTOM_LEFT: Vec2(-inv_sqrt2, inv_sqrt2),
        # Triângulo com vértice no bottom-right
        TriangleOrientation.BOTTOM_RIGHT: Vec2(inv_sqrt2, inv_sqrt2),
    }
    return normals.get(orientation, Vec2(0, -1))


def _point_in_triangle(px: float, py: float, vertices: np.ndarray) -> bool:
    """Checa se ponto está dentro do triângulo usando cross products."""
    if len(vertices) < 3:
        return False

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    v = vertices
    d1 = sign((px, py), v[0], v[1])
    d2 = sign((px, py), v[1], v[2])
    d3 = sign((px, py), v[2], v[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def _collide_triangle(
    pos: Vec2, vel: Vec2, radius: float, block: Block
) -> Optional[Vec2]:
    """
    Colisão com bloco triangular.
    Primeiro checa AABB, depois refina com a diagonal.
    """
    # Checa AABB primeiro (fast reject)
    aabb_normal = _collide_aabb(pos, vel, radius, block.x, block.y, block.w, block.h)
    if aabb_normal is None:
        return None

    # Se o ponto está perto da região do triângulo, usa a normal diagonal
    # Caso contrário, usa a normal AABB das bordas retas
    if (
        block.tri_orientation != TriangleOrientation.UNKNOWN
        and len(block.vertices) >= 3
    ):
        # Checa se a bola está mais perto da hipotenusa
        diag_normal = _get_triangle_diagonal_normal(block.tri_orientation)

        # Projeção simples: se a bola está "do lado" da diagonal, usa ela
        center_to_ball = Vec2(pos.x - block.cx, pos.y - block.cy)
        if abs(center_to_ball.dot(diag_normal)) < block.w * 0.7:
            return diag_normal

    return aabb_normal


# ─────────────────────────────────────────────
# COLISÃO COM POWER-UP (circular)
# ─────────────────────────────────────────────


def _collide_powerup(pos: Vec2, radius: float, pu: PowerUp) -> bool:
    """Checa se a bola colidiu com o power-up."""
    dx = pos.x - pu.cx
    dy = pos.y - pu.cy
    dist = math.sqrt(dx * dx + dy * dy)
    return dist < (radius + pu.radius)


def _apply_laser(
    pu: PowerUp,
    blocks: List[Block],
    block_hp: dict,
    ball_count: int,
) -> Tuple[float, int]:
    """
    Efeito do laser (<>): causa ball_count de dano a todos os blocos
    na mesma linha (Y) e coluna (X) do power-up.
    Retorna (score_ganho, blocos_destruidos).
    """
    score = 0.0
    destroyed = 0
    half_cell = cfg.BLOCK_SIZE / 2

    for block in blocks:
        if block_hp.get(id(block), 0) <= 0:
            continue

        in_row = abs(block.cy - pu.cy) < half_cell
        in_col = abs(block.cx - pu.cx) < half_cell
        if not (in_row or in_col):
            continue

        block_hp[id(block)] -= ball_count
        score += cfg.SCORE_HIT_BLOCK * ball_count

        if block.cy >= cfg.DANGER_ZONE_Y:
            danger_ratio = (block.cy - cfg.DANGER_ZONE_Y) / max(
                1, cfg.WALL_BOTTOM - cfg.DANGER_ZONE_Y
            )
            score += cfg.SCORE_DANGER_HIT * (1 + danger_ratio * 3)

        if block_hp[id(block)] <= 0:
            destroyed += 1
            score += cfg.SCORE_DESTROY_BLOCK
            if block.cy >= cfg.DANGER_ZONE_Y:
                danger_ratio = (block.cy - cfg.DANGER_ZONE_Y) / max(
                    1, cfg.WALL_BOTTOM - cfg.DANGER_ZONE_Y
                )
                score += cfg.SCORE_DANGER_DESTROY * (1 + danger_ratio * 2)

    return score, destroyed


# ─────────────────────────────────────────────
# MULTIPLICADOR DE PRIORIDADE POR FILEIRA
# ─────────────────────────────────────────────


def get_row_multiplier(block_y: float) -> float:
    """Retorna o multiplicador de pontuação com base na fileira do bloco (cfg.LEVELS)."""
    # Abaixo do último nível = jogo perdido se não destruir
    if block_y >= cfg.LEVELS[0]:
        return 30.0
    for idx, row_y in enumerate(cfg.LEVELS):
        if abs(block_y - row_y) < 25:
            return cfg.ROW_MULTIPLIERS.get(idx, 1.0)
    # Fallback interpolado por posição
    if block_y > cfg.LEVELS[1]: return 5.0
    if block_y > cfg.LEVELS[3]: return 1.5
    return 1.0


# ─────────────────────────────────────────────
# SIMULAÇÃO PRINCIPAL
# ─────────────────────────────────────────────


def simulate_trajectory(
    start_x: float,
    start_y: float,
    angle_deg: float,
    blocks: List[Block],
    powerups: List[PowerUp],
    ball_count: int = 1,
    record_path: bool = False,
) -> SimResult:
    """
    Simula a trajetória de uma bola disparada do ponto (start_x, start_y)
    no ângulo dado (0° = direita, 90° = cima, 180° = esquerda).

    No BBTAN todas as bolas seguem a mesma trajetória, então cada ponto
    de colisão causa ball_count hits no bloco (uma vez por bola).

    Retorna SimResult com a pontuação e estatísticas.
    """
    # Converte ângulo para vetor de direção
    # No BBTAN, 90° = direto pra cima, >90 vai pra esquerda, <90 pra direita
    angle_rad = math.radians(angle_deg)
    vel = Vec2(math.cos(angle_rad), -math.sin(angle_rad))
    vel = vel.normalized()
    vel = vel * cfg.BALL_SPEED

    pos = Vec2(start_x, start_y)
    radius = cfg.BALL_RADIUS

    # Limite inferior: o maior entre WALL_BOTTOM e a posição real da bola + margem
    # Isso garante que a simulação funciona mesmo se WALL_BOTTOM estiver errado
    effective_bottom = max(cfg.WALL_BOTTOM, start_y + radius + 2)

    # Estado da simulação
    score = 0.0
    total_hits = 0
    blocks_destroyed = 0
    powerups_collected = 0
    extra_balls_collected = 0
    path = []

    # Cópia local do HP dos blocos para esta simulação
    # Todas as N bolas passam pelo mesmo ponto, então cada colisão
    # causa ball_count de dano
    block_hp = {id(b): b.hp for b in blocks}
    # Tracking de power-ups já coletados
    collected_pus = set()
    # Tracking de bounces consecutivos em blocos (para bônus de multi-bounce)
    consecutive_block_hits = 0
    last_hit_id = None
    went_up = False  # flag: bola já subiu pelo menos uma vez

    for step in range(cfg.MAX_SIM_STEPS):
        # Avança a posição
        pos = pos + vel * (cfg.SIM_STEP_SIZE / cfg.BALL_SPEED)

        if record_path and step % 4 == 0:
            path.append(pos.as_tuple())

        # Marca que a bola subiu (para não terminar no primeiro frame)
        if pos.y < start_y - radius * 2:
            went_up = True

        # ── Colisão com paredes laterais ──
        if pos.x - radius <= cfg.WALL_LEFT:
            pos.x = cfg.WALL_LEFT + radius
            vel.x = abs(vel.x)
            consecutive_block_hits = 0

        elif pos.x + radius >= cfg.WALL_RIGHT:
            pos.x = cfg.WALL_RIGHT - radius
            vel.x = -abs(vel.x)
            consecutive_block_hits = 0

        # ── Colisão com teto ──
        if pos.y - radius <= cfg.WALL_TOP:
            pos.y = cfg.WALL_TOP + radius
            vel.y = abs(vel.y)
            consecutive_block_hits = 0

        # ── Bola saiu por baixo (fim do turno) ──
        # Só conta como saída se a bola já subiu pelo menos uma vez
        if went_up and pos.y + radius >= effective_bottom:
            break

        # ── Colisão com blocos ──
        for block in blocks:
            if block_hp.get(id(block), 0) <= 0:
                continue  # bloco já destruído nesta simulação

            # Escolhe método de colisão baseado no tipo
            if block.block_type == BlockType.TRIANGLE:
                normal = _collide_triangle(pos, vel, radius, block)
            elif block.w > cfg.BLOCK_SIZE * 1.5:  # Identifica se é o bloco grandão pela largura
                normal = _collide_aabb2(pos, vel, radius, block.x, block.y, block.w, block.h)
            else:
                normal = _collide_aabb(pos, vel, radius, block.x, block.y, block.w, block.h)

            if normal is not None:
                # Reflete a velocidade
                vel_new = vel.reflect(normal)
                vel.x = vel_new.x
                vel.y = vel_new.y

                # Empurra a bola pra fora do bloco
                pos = pos + normal * (radius * 0.5)

                # Dano = ball_count (todas as bolas batem nesse ponto)
                damage = ball_count
                block_hp[id(block)] -= damage
                total_hits += 1
                score += cfg.SCORE_HIT_BLOCK * ball_count

                # ── Bônus zona de perigo ──
                # Blocos mais pra baixo ganham prioridade crescente
                if block.cy >= cfg.DANGER_ZONE_Y:
                    # Quanto mais embaixo, maior o bônus (0.0 a 1.0)
                    danger_ratio = (block.cy - cfg.DANGER_ZONE_Y) / max(
                        1, cfg.WALL_BOTTOM - cfg.DANGER_ZONE_Y
                    )
                    score += cfg.SCORE_DANGER_HIT * (1 + danger_ratio * 3)

                    # Última fileira: quase game over
                    if block.cy >= cfg.WALL_BOTTOM - cfg.BLOCK_SIZE:
                        score += cfg.SCORE_CRITICAL_ROW

                # Multi-bounce bonus
                bid = id(block)
                if last_hit_id is not None and last_hit_id != bid:
                    consecutive_block_hits += 1
                    if consecutive_block_hits >= 2:
                        score += cfg.SCORE_MULTI_BOUNCE * consecutive_block_hits
                else:
                    consecutive_block_hits = 1
                last_hit_id = bid

                # Bloco destruído?
                if block_hp[id(block)] <= 0:
                    blocks_destroyed += 1
                    score += cfg.SCORE_DESTROY_BLOCK

                    # Bônus extra por destruir bloco em perigo
                    if block.cy >= cfg.DANGER_ZONE_Y:
                        danger_ratio = (block.cy - cfg.DANGER_ZONE_Y) / max(
                            1, cfg.WALL_BOTTOM - cfg.DANGER_ZONE_Y
                        )
                        score += cfg.SCORE_DANGER_DESTROY * (1 + danger_ratio * 2)

                break  # só processa uma colisão com bloco por step

        # ── Colisão com power-ups ──
        for i, pu in enumerate(powerups):
            if i in collected_pus:
                continue
            if _collide_powerup(pos, radius, pu):
                collected_pus.add(i)
                powerups_collected += 1

                if pu.power_up_type == PowerUpType.LASER:
                    laser_score, laser_destroyed = _apply_laser(pu, blocks, block_hp, ball_count)
                    score += laser_score
                    blocks_destroyed += laser_destroyed
                elif pu.power_up_type == PowerUpType.EXTRA_BALL:
                    extra_balls_collected += 1
                    score += cfg.SCORE_HIT_POWERUP
                else:
                    score += cfg.SCORE_HIT_POWERUP

    # Bônus por loop: se bateu muitas vezes em blocos diferentes
    if total_hits > 10:
        score += cfg.SCORE_LOOP_BONUS * (total_hits // 10)

    return SimResult(
        angle=angle_deg,
        score=score,
        total_hits=total_hits,
        blocks_destroyed=blocks_destroyed,
        powerups_collected=powerups_collected,
        extra_balls_collected=extra_balls_collected,
        path=path if record_path else [],
    )


# ─────────────────────────────────────────────
# BUSCA DO MELHOR ÂNGULO
# ─────────────────────────────────────────────


def find_best_angle(
    ball_x: float,
    ball_y: float,
    blocks: List[Block],
    powerups: List[PowerUp],
    ball_count: int,
    angle_min: int = cfg.ANGLE_MIN,
    angle_max: int = cfg.ANGLE_MAX,
    angle_step: int = cfg.ANGLE_STEP,
) -> Tuple[SimResult, List[SimResult]]:
    """
    Testa todos os ângulos no range e retorna o melhor.
    ball_count: número de bolas atual (afeta dano por colisão).
    Returns: (melhor_resultado, lista_todos_resultados)
    """
    all_results: List[SimResult] = []

    for angle in range(angle_min, angle_max + 1, angle_step):
        result = simulate_trajectory(
            ball_x,
            ball_y,
            float(angle),
            blocks,
            powerups,
            ball_count=ball_count,
            record_path=False,
        )
        all_results.append(result)

    # Ordena por score decrescente
    all_results.sort(key=lambda r: r.score, reverse=True)

    best = all_results[0]

    # Re-simula o melhor com path para debug
    best_with_path = simulate_trajectory(
        ball_x,
        ball_y,
        best.angle,
        blocks,
        powerups,
        ball_count=ball_count,
        record_path=True,
    )

    return best_with_path, all_results


if __name__ == "__main__":
    # Teste rápido com blocos fictícios
    test_blocks = [
        Block(
            x=100, y=50, w=50, h=50, cx=125, cy=75, block_type=BlockType.SQUARE, hp=2
        ),
        Block(
            x=200, y=100, w=50, h=50, cx=225, cy=125, block_type=BlockType.SQUARE, hp=1
        ),
    ]
    test_pus = [PowerUp(cx=350, cy=80, radius=12)]

    best, all_res = find_best_angle(
        ball_x=225,
        ball_y=550,
        blocks=test_blocks,
        powerups=test_pus,
    )

    print(f"Melhor angulo: {best.angle}°")
    print(f"  Score: {best.score}")
    print(f"  Hits: {best.total_hits}")
    print(f"  Destruidos: {best.blocks_destroyed}")
    print(f"  Power-ups: {best.powerups_collected}")
    print(f"\nTop 5 angulos:")
    for r in all_res[:5]:
        print(
            f"  {r.angle:6.1f}° -> score={r.score:.0f}  hits={r.total_hits}  destroyed={r.blocks_destroyed}"
        )
