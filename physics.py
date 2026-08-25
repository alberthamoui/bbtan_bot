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
class ScoreTerms:
    """
    Componentes CRUS de uma simulação, sem pesos aplicados.

    Existe para separar "o que aconteceu no turno" de "quanto isso vale": a
    contabilidade acumula aqui e compute_score() aplica os pesos do config uma
    única vez. Antes havia dois acumuladores paralelos (score e weighted_hits)
    calculando quase a mesma coisa com pesos diferentes, e só o segundo decidia.
    """

    weighted_hits: float = 0.0  # Σ ROW_MULTIPLIER × dano efetivo
    weighted_destroys: float = 0.0  # Σ ROW_MULTIPLIER por bloco destruído
    extra_balls: int = 0
    bonus_b: int = 0
    total_hits: int = 0
    danger_hits: int = 0  # hits em rows 0-1 — diagnóstico, fora do ranking
    row0_surviving_hp: int = 0  # HP vivo na fileira crítica ao fim do turno

    def add(self, other: "ScoreTerms") -> None:
        """Acumula os termos de outra simulação parcial (ex: efeito do laser)."""
        self.weighted_hits += other.weighted_hits
        self.weighted_destroys += other.weighted_destroys
        self.extra_balls += other.extra_balls
        self.bonus_b += other.bonus_b
        self.total_hits += other.total_hits
        self.danger_hits += other.danger_hits
        self.row0_surviving_hp += other.row0_surviving_hp


def compute_score(t: ScoreTerms) -> float:
    """
    Nota final de um ângulo — o ÚNICO critério de ordenação de find_best_angle.

    Cada peso tem um trabalho só. W_ROW0_SURVIVE é o único termo que olha o
    estado final do tabuleiro (HP vivo na fileira que causa game over); os
    outros contam o que a bola fez no caminho.
    """
    return (
        cfg.W_HIT * t.weighted_hits
        + cfg.W_DESTROY * t.weighted_destroys
        + cfg.W_EXTRA_BALL * t.extra_balls
        + cfg.W_BONUS_B * t.bonus_b
        + cfg.W_LOOP * (t.total_hits // 10)
        - cfg.W_ROW0_SURVIVE * t.row0_surviving_hp
    )


@dataclass
class SimResult:
    """Resultado de uma simulação de trajetória."""

    angle: float
    score: float  # critério de ranking (compute_score dos ScoreTerms)
    total_hits: int
    weighted_hits: float  # hits ponderados por ROW_MULTIPLIERS
    danger_hits: int  # hits efetivos em blocos das rows 0-1 (ranking primário)
    blocks_destroyed: int
    powerups_collected: int
    extra_balls_collected: int = 0
    path: List[Tuple[float, float]] = field(default_factory=list)
    collisions: List[Tuple[float, float]] = field(default_factory=list)
    # Um path/collisions POR BOLA lancada (so com record_path=True). Bolas
    # posteriores seguem trajetorias re-simuladas apos blocos destruidos —
    # usado pelo animate_dryrun pra nao repetir o path da 1a bola em todas.
    ball_paths: List[List[Tuple[float, float]]] = field(default_factory=list)
    ball_collisions: List[List[Tuple[float, float]]] = field(default_factory=list)
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
# COLISÃO COM TRIÂNGULO
# ─────────────────────────────────────────────


def _get_triangle_diagonal_normal(orientation: TriangleOrientation) -> Vec2:
    """
    Retorna a normal outward da hipotenusa (aponta da superfície para a área vazia).
    TOP_LEFT  (ângulo reto em cima-esq): corpo em cima-esq, vazio em baixo-dir → (+1,+1)
    TOP_RIGHT (ângulo reto em cima-dir): corpo em cima-dir, vazio em baixo-esq → (-1,+1)
    BOT_LEFT  (ângulo reto em baixo-esq): corpo em baixo-esq, vazio em cima-dir → (+1,-1)
    BOT_RIGHT (ângulo reto em baixo-dir): corpo em baixo-dir, vazio em cima-esq → (-1,-1)
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    normals = {
        TriangleOrientation.TOP_LEFT: Vec2(inv_sqrt2, inv_sqrt2),
        TriangleOrientation.TOP_RIGHT: Vec2(-inv_sqrt2, inv_sqrt2),
        TriangleOrientation.BOTTOM_LEFT: Vec2(inv_sqrt2, -inv_sqrt2),
        TriangleOrientation.BOTTOM_RIGHT: Vec2(-inv_sqrt2, -inv_sqrt2),
    }
    return normals.get(orientation, Vec2(0, -1))


def _dist_to_segment_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-8:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def _point_near_polygon(px: float, py: float, radius: float, verts: np.ndarray) -> bool:
    """True se (px,py) está dentro do polígono ou a menos de radius de alguma aresta."""
    n = len(verts)
    if n < 3:
        return True

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(verts[i][0]), float(verts[i][1])
        xj, yj = float(verts[j][0]), float(verts[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    if inside:
        return True

    r_sq = radius * radius
    j = n - 1
    for i in range(n):
        xi, yi = float(verts[i][0]), float(verts[i][1])
        xj, yj = float(verts[j][0]), float(verts[j][1])
        if _dist_to_segment_sq(px, py, xi, yi, xj, yj) < r_sq:
            return True
        j = i
    return False


def _collide_triangle(
    pos: Vec2, vel: Vec2, radius: float, block: Block
) -> Optional[Vec2]:
    """
    Colisão com bloco triangular.
    AABB como fast-reject, depois verifica proximidade ao polígono real.
    Distingue entrada pelo cateto (rebate como parede) vs hipotenusa
    (reflexão pela normal diagonal).
    """
    aabb_normal = _collide_aabb(pos, vel, radius, block.x, block.y, block.w, block.h)
    if aabb_normal is None:
        return None

    if len(block.vertices) >= 3:
        if not _point_near_polygon(pos.x, pos.y, radius, block.vertices):
            return None

    if block.tri_orientation == TriangleOrientation.UNKNOWN:
        return aabb_normal

    diag_normal = _get_triangle_diagonal_normal(block.tri_orientation)
    d = Vec2(pos.x - block.cx, pos.y - block.cy)

    if d.dot(diag_normal) > 0:
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
) -> Tuple[ScoreTerms, int]:
    """
    Efeito do laser: LASER_H ataca blocos na mesma linha (Y),
    LASER_V ataca blocos na mesma coluna (X).
    Retorna (termos, blocos_destruidos).

    Usa a mesma contabilidade dos hits de bola (dano efetivo limitado ao HP
    restante, ROW_MULTIPLIERS por fileira) para que o laser conte no ranking
    de find_best_angle.
    """
    terms = ScoreTerms()
    destroyed = 0
    half_cell = cfg.BLOCK_SIZE / 2

    for block in blocks:
        hp = block_hp.get(id(block), 0)
        if hp <= 0:
            continue

        if pu.power_up_type == PowerUpType.LASER_H:
            hit = abs(block.cy - pu.cy) < half_cell
        else:
            hit = abs(block.cx - pu.cx) < half_cell
        if not hit:
            continue

        effective = min(ball_count, hp)
        block_hp[id(block)] -= ball_count
        row_mult = cfg.ROW_MULTIPLIERS.get(block.row_index, 1.0)
        terms.total_hits += effective
        terms.weighted_hits += row_mult * effective
        if block.row_index <= 1:
            terms.danger_hits += effective

        if block_hp[id(block)] <= 0:
            destroyed += 1
            terms.weighted_destroys += row_mult

    return terms, destroyed


# ─────────────────────────────────────────────
# SIMULAÇÃO PRINCIPAL (sequencial por bola)
# ─────────────────────────────────────────────


def _sim_one_ball(
    start_x: float,
    start_y: float,
    angle_deg: float,
    blocks: List[Block],
    block_hp: dict,
    powerups: List[PowerUp],
    collected_pus: set,
    record_path: bool = False,
) -> Tuple[
    List[Tuple[int, int]], set, List[Tuple[float, float]], List[Tuple[float, float]]
]:
    """
    Simula UMA bola (1 dano por colisão). Não altera block_hp externo.
    Retorna (hit_sequence, new_pu_indices, path, collisions).
    hit_sequence: [(block_id, row_index), ...] em ordem de colisão.
    collisions: posicoes onde a bola colidiu (paredes, blocos, triangulos).
      So eh populado quando record_path=True.
    """
    angle_rad = math.radians(angle_deg)
    vel = Vec2(math.cos(angle_rad), -math.sin(angle_rad))
    vel = vel.normalized() * cfg.BALL_SPEED
    pos = Vec2(start_x, start_y)
    radius = cfg.BALL_RADIUS
    effective_bottom = max(cfg.WALL_BOTTOM, start_y + radius + 2)

    local_hp = dict(block_hp)
    hit_seq: List[Tuple[int, int]] = []
    new_pus: set = set()
    path: List[Tuple[float, float]] = []
    collisions: List[Tuple[float, float]] = []
    went_up = False

    # MAX_SIM_STEPS é derivado de SIM_PATH_BUDGET / SIM_STEP_SIZE, então o teto do
    # loop é sempre o mesmo alcance em pixels, independente da resolução escolhida.
    for step in range(cfg.MAX_SIM_STEPS):
        # BALL_SPEED se cancela: vel já foi normalizado e multiplicado por ele acima,
        # então o avanço real é sempre `direção × SIM_STEP_SIZE`.
        pos = pos + vel * (cfg.SIM_STEP_SIZE / cfg.BALL_SPEED)

        if record_path and step % 4 == 0:
            path.append(pos.as_tuple())

        if pos.y < start_y - radius * 2:
            went_up = True

        if pos.x - radius <= cfg.WALL_LEFT:
            pos.x = cfg.WALL_LEFT + radius
            vel.x = abs(vel.x)
            if record_path:
                collisions.append(pos.as_tuple())
        elif pos.x + radius >= cfg.WALL_RIGHT:
            pos.x = cfg.WALL_RIGHT - radius
            vel.x = -abs(vel.x)
            if record_path:
                collisions.append(pos.as_tuple())

        if pos.y - radius <= cfg.WALL_TOP:
            pos.y = cfg.WALL_TOP + radius
            vel.y = abs(vel.y)
            if record_path:
                collisions.append(pos.as_tuple())

        if went_up and pos.y + radius >= effective_bottom:
            break

        for block in blocks:
            if local_hp.get(id(block), 0) <= 0:
                continue

            if block.block_type == BlockType.TRIANGLE:
                normal = _collide_triangle(pos, vel, radius, block)
            else:
                normal = _collide_aabb(
                    pos, vel, radius, block.x, block.y, block.w, block.h
                )

            if normal is not None:
                if record_path:
                    collisions.append(pos.as_tuple())
                vel_new = vel.reflect(normal)
                vel.x = vel_new.x
                vel.y = vel_new.y
                pos = pos + normal * (radius * 0.5)
                local_hp[id(block)] -= 1
                hit_seq.append((id(block), block.row_index))
                break

        for i, pu in enumerate(powerups):
            if i in collected_pus or i in new_pus:
                continue
            if _collide_powerup(pos, radius, pu):
                new_pus.add(i)

    return hit_seq, new_pus, path, collisions


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
    Simula ball_count bolas sequencialmente. Cada bola causa 1 dano por
    colisão. Blocos destruídos por bolas anteriores são removidos para
    bolas posteriores, alterando a trajetória.

    Otimização: agrupa bolas que seguem a mesma trajetória (batch) até
    que o próximo bloco seja destruído, evitando re-simular cada bola.
    """
    block_hp = {id(b): b.hp for b in blocks}
    block_map = {id(b): b for b in blocks}
    collected_pus: set = set()

    terms = ScoreTerms()
    total_destroyed = 0
    total_pu = 0
    first_path: List[Tuple[float, float]] = []
    first_collisions: List[Tuple[float, float]] = []
    ball_paths: List[List[Tuple[float, float]]] = []
    ball_collisions: List[List[Tuple[float, float]]] = []

    remaining = ball_count

    while remaining > 0:
        active = [b for b in blocks if block_hp.get(id(b), 0) > 0]

        hit_seq, new_pus, path, collisions = _sim_one_ball(
            start_x,
            start_y,
            angle_deg,
            active,
            block_hp,
            powerups,
            collected_pus,
            record_path=record_path,
        )

        if record_path and not first_path:
            first_path = path
            first_collisions = collisions

        # ── Power-ups (coletados pela primeira bola do batch) ──
        laser_fired = False
        for pi in new_pus:
            collected_pus.add(pi)
            total_pu += 1
            pu = powerups[pi]
            if pu.power_up_type in (PowerUpType.LASER_H, PowerUpType.LASER_V):
                laser_terms, laser_destroyed = _apply_laser(pu, blocks, block_hp, 1)
                terms.add(laser_terms)
                total_destroyed += laser_destroyed
                laser_fired = True
            elif pu.power_up_type == PowerUpType.EXTRA_BALL:
                terms.extra_balls += 1
            elif pu.power_up_type == PowerUpType.BONUS_B:
                terms.bonus_b += 1

        if not hit_seq:
            # Nada mais pra acertar — as bolas restantes seguem esta mesma
            # trajetoria (nenhum bloco muda de estado daqui pra frente).
            if record_path:
                ball_paths.extend([path] * remaining)
                ball_collisions.extend([collisions] * remaining)
            break

        # ── Batch: quantas bolas seguem esta mesma trajetória? ──
        hits_per_block: dict = {}
        row_of: dict = {}
        for bid, ridx in hit_seq:
            hits_per_block[bid] = hits_per_block.get(bid, 0) + 1
            row_of[bid] = ridx

        if laser_fired or record_path:
            # record_path: simula 1 bola por voo pra gravar o path exato de
            # CADA bola (o batch reutilizaria o path da 1a bola para o batch
            # inteiro, inclusive apos o bloco quebrar). So roda no re-simulate
            # do melhor angulo, entao o custo extra e desprezivel.
            batch = 1
        else:
            batch = remaining
            for bid, nh in hits_per_block.items():
                hp = block_hp.get(bid, 0)
                if hp > 0 and nh > 0:
                    batch = min(batch, math.ceil(hp / nh))
            batch = max(1, batch)

        if record_path:
            ball_paths.extend([path] * batch)
            ball_collisions.extend([collisions] * batch)

        # ── Aplica dano do batch ──
        for bid, nh in hits_per_block.items():
            hp = block_hp.get(bid, 0)
            if hp <= 0:
                continue
            damage = batch * nh
            effective = min(damage, hp)
            block_hp[bid] -= damage

            rm = cfg.ROW_MULTIPLIERS.get(row_of[bid], 1.0)
            terms.total_hits += effective
            terms.weighted_hits += rm * effective
            if row_of[bid] <= 1:
                terms.danger_hits += effective

            if block_hp[bid] <= 0:
                total_destroyed += 1
                terms.weighted_destroys += rm

        remaining -= batch

    # ── Estado FINAL: HP que sobrou vivo na fileira crítica ──
    for bid, hp in block_hp.items():
        if hp > 0:
            b = block_map.get(bid)
            if b is not None and b.row_index == 0:
                terms.row0_surviving_hp += hp

    return SimResult(
        angle=angle_deg,
        score=compute_score(terms),
        total_hits=terms.total_hits,
        weighted_hits=terms.weighted_hits,
        danger_hits=terms.danger_hits,
        blocks_destroyed=total_destroyed,
        powerups_collected=total_pu,
        extra_balls_collected=terms.extra_balls,
        path=first_path if record_path else [],
        collisions=first_collisions if record_path else [],
        ball_paths=ball_paths if record_path else [],
        ball_collisions=ball_collisions if record_path else [],
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
    progress_callback=None,
) -> Tuple[SimResult, List[SimResult]]:
    """
    Simula todos os ângulos de 1 em 1 grau.
    Returns: (melhor_resultado, lista_todos_resultados)

    progress_callback(feitos, total): chamado a cada ângulo simulado, e mais uma
    vez ao fim da re-simulação do melhor ângulo. Opcional — physics não imprime
    nada por conta própria; quem chama decide como (ou se) mostra o progresso.
    O total inclui a re-simulação final, entao a barra so chega a 100% quando a
    jogada esta de fato decidida.

    Progresso e contado por ANGULO, sem tentar pesar o custo de cada um. Uma
    tentativa de pesar por 1/sin(angulo) (trajetoria rasa percorre mais) foi
    medida e saiu PIOR que a contagem simples: 26.6% de erro contra 21.4%. O
    custo real varia 100x entre angulos (0.9ms a 90ms) e depende de quantos
    blocos a bola encontra antes de voltar a base — ou seja, do tabuleiro, nao
    do angulo. Nenhum modelo estatico acerta isso; quem chama que estime o
    tempo restante pela media corrida, que se auto-corrige.
    """
    angles = list(range(angle_min, angle_max + 1))
    total_steps = len(angles) + 1  # +1 = re-simulação com path

    all_results = []
    for i, a in enumerate(angles):
        r = simulate_trajectory(
            ball_x,
            ball_y,
            float(a),
            blocks,
            powerups,
            ball_count=ball_count,
            record_path=False,
        )
        all_results.append(r)
        if progress_callback is not None:
            progress_callback(i + 1, total_steps)

    # Ordenação única por score. Antes havia dois modos: com bloco em row 0-1, o
    # desempate lexicográfico por danger_hits vinha primeiro — o que fazia um tiro
    # que belisca 3x um bloco de 50 HP da fileira crítica vencer um que a destrói e
    # ainda limpa 5 outros. Os ROW_MULTIPLIERS (row0=100x) já codificam esse perigo
    # de forma contínua, então a camada extra era redundante e prejudicial.
    all_results.sort(key=lambda r: (r.score, r.total_hits), reverse=True)
    best = all_results[0]

    best_with_path = simulate_trajectory(
        ball_x,
        ball_y,
        best.angle,
        blocks,
        powerups,
        ball_count=ball_count,
        record_path=True,
    )
    if progress_callback is not None:
        progress_callback(total_steps, total_steps)

    return best_with_path, all_results
