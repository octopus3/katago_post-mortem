"""复盘逻辑 — 从目数、厚薄、死活三个维度分析问题手

核心原则（与实际棋评一致）:
    1. 目数/实地得失  — 最高优先级
    2. 厚薄/势力消长  — 第二优先级
    3. 死活安危        — 必须关注

用法:
    reviewer = Reviewer(engine, cfg.review, llm)
    result = await reviewer.review_game(parsed_game)
    print(result.total_problems, "个问题手")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from config import ReviewConfig
from katago_engine import KataGoEngine
from llm_service import LLMService
from sgf_parser import ParsedGame

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int], Coroutine]]

# ── 严重程度 ──────────────────────────────────────────────────


class Severity(str, Enum):
    """问题手严重程度。"""
    MINOR = "minor"               # 小疑问手 2–5 %
    QUESTIONABLE = "questionable"  # 疑问手   5–10 %
    BAD = "bad"                    # 恶手     10 %+


_SEVERITY_THRESHOLDS: List[Tuple[float, Severity]] = [
    (0.10, Severity.BAD),
    (0.05, Severity.QUESTIONABLE),
    (0.02, Severity.MINOR),
]

_SEVERITY_CN: Dict[Severity, str] = {
    Severity.MINOR: "小疑问手",
    Severity.QUESTIONABLE: "疑问手",
    Severity.BAD: "恶手",
}


def _classify_severity(wr_drop: float) -> Optional[Severity]:
    """根据胜率跌幅返回严重程度等级，低于 2% 返回 None。"""
    for threshold, sev in _SEVERITY_THRESHOLDS:
        if wr_drop >= threshold:
            return sev
    return None


# ── 棋盘区域定义（19 路） ────────────────────────────────────

_REGIONS_19: Dict[str, Tuple[range, range]] = {
    "左上角": (range(0, 7), range(0, 7)),
    "上边":   (range(0, 7), range(7, 12)),
    "右上角": (range(0, 7), range(12, 19)),
    "左边":   (range(7, 12), range(0, 7)),
    "中腹":   (range(7, 12), range(7, 12)),
    "右边":   (range(7, 12), range(12, 19)),
    "左下角": (range(12, 19), range(0, 7)),
    "下边":   (range(12, 19), range(7, 12)),
    "右下角": (range(12, 19), range(12, 19)),
}

# ownership 值分级阈值
_OWN_CONFIRMED = 0.80   # 确定目数
_OWN_LIKELY = 0.50       # 大致目数
_OWN_INFLUENCE = 0.20    # 势力/厚薄区
_LIFE_DEATH_FLIP = 0.60  # 归属骤变阈值（可能涉及死活）
_REGION_CHANGE_MIN = 0.05  # 区域变化值得报告的最小幅度


# ── 数据结构 ──────────────────────────────────────────────────


@dataclass
class Variation:
    """KataGo 推荐的单个候选变化。"""
    move: str
    winrate: float
    score_lead: float
    visits: int
    pv: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "move": self.move,
            "winrate": round(self.winrate, 4),
            "scoreLead": round(self.score_lead, 2),
            "visits": self.visits,
            "pv": self.pv,
        }


@dataclass
class TerritoryEstimate:
    """基于 ownership 的目数估算。"""
    black_territory: float
    white_territory: float

    @property
    def diff(self) -> float:
        """黑方净目数（正=黑优）。"""
        return self.black_territory - self.white_territory

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blackTerritory": round(self.black_territory, 1),
            "whiteTerritory": round(self.white_territory, 1),
            "diff": round(self.diff, 1),
        }


@dataclass
class MoveAnalysis:
    """单手的完整分析结果。"""
    move_number: int
    color: str                                         # "B" / "W"
    gtp_coord: str                                     # 实战着点
    winrate_before: float                              # 落子前行棋方胜率
    winrate_after: float                               # 落子后行棋方胜率
    winrate_drop: float                                # 胜率跌幅（正值=变差）
    score_before: float                                # 落子前行棋方目差
    score_after: float                                 # 落子后行棋方目差
    score_drop: float                                  # 目差跌幅
    black_winrate: float                               # 黑方视角胜率（胜率曲线用）
    best_variations: List[Variation] = field(default_factory=list)
    is_problem: bool = False
    severity: Optional[Severity] = None
    comment: str = ""
    # ownership 分析（需要 includeOwnership=True）
    territory_before: Optional[TerritoryEstimate] = None
    territory_after: Optional[TerritoryEstimate] = None
    ownership_context: str = ""                        # 目数/厚薄/死活分析文本

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "moveNumber": self.move_number,
            "color": self.color,
            "gtpCoord": self.gtp_coord,
            "winrateBefore": round(self.winrate_before, 4),
            "winrateAfter": round(self.winrate_after, 4),
            "winrateDrop": round(self.winrate_drop, 4),
            "scoreBefore": round(self.score_before, 2),
            "scoreAfter": round(self.score_after, 2),
            "scoreDrop": round(self.score_drop, 2),
            "blackWinrate": round(self.black_winrate, 4),
            "bestVariations": [v.to_dict() for v in self.best_variations],
            "isProblem": self.is_problem,
            "severity": self.severity.value if self.severity else None,
            "comment": self.comment,
        }
        if self.territory_before:
            d["territoryBefore"] = self.territory_before.to_dict()
        if self.territory_after:
            d["territoryAfter"] = self.territory_after.to_dict()
        return d


@dataclass
class ReviewResult:
    """一盘棋的完整复盘结果。"""
    game_info: Dict[str, Any]
    move_analyses: List[MoveAnalysis]
    problem_moves: List[MoveAnalysis]
    black_winrate_curve: List[float]
    initial_black_winrate: float = 0.5

    @property
    def total_moves(self) -> int:
        return len(self.move_analyses)

    @property
    def total_problems(self) -> int:
        return len(self.problem_moves)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gameInfo": self.game_info,
            "totalMoves": self.total_moves,
            "totalProblems": self.total_problems,
            "initialBlackWinrate": round(self.initial_black_winrate, 4),
            "moveAnalyses": [m.to_dict() for m in self.move_analyses],
            "problemMoves": [
                {"moveNumber": m.move_number, "severity": m.severity.value if m.severity else None}
                for m in self.problem_moves
            ],
            "blackWinrateCurve": [round(w, 4) for w in self.black_winrate_curve],
        }


# ── 复盘引擎 ─────────────────────────────────────────────────


class Reviewer:
    """协调 KataGo 分析、胜率落差计算、问题手识别与 LLM 解说。

    解说维度优先级: 目数 > 厚薄 > 死活

    典型流程:
        1. 逐手发送 KataGo 分析请求（含 ownership）
        2. 对比前后胜率 + ownership 计算落差
        3. 按阈值标记问题手并分级
        4. 从 ownership 提取目数得失、厚薄变化、死活信号
        5. 组装结构化 prompt，调用 LLM 生成分维度解说
    """

    def __init__(
        self,
        engine: KataGoEngine,
        review_cfg: ReviewConfig,
        llm: Optional[LLMService] = None,
    ):
        self._engine = engine
        self._cfg = review_cfg
        self._llm = llm

    # ── 主入口 ────────────────────────────────────────────────

    async def review_game(
        self,
        game: ParsedGame,
        *,
        with_comments: bool = True,
        progress_callback: ProgressCallback = None,
    ) -> ReviewResult:
        """对一盘棋进行完整复盘。

        Args:
            game: sgf_parser.parse_sgf() 返回的 ParsedGame
            with_comments: 是否为问题手生成 LLM 解说
            progress_callback: ``async callable(current, total)`` 进度回调

        Returns:
            ReviewResult 包含每手分析、问题手列表、胜率曲线等
        """
        logger.info(
            "开始复盘: %s vs %s (%d 手)",
            game.info.black_player, game.info.white_player, game.total_moves,
        )

        raw = await self._analyze_all_positions(game, progress_callback)

        initial_root = raw[0].get("rootInfo", {})
        initial_black_wr = initial_root.get("winrate", 0.5)

        move_analyses = self._compute_move_analyses(game, raw)
        problem_moves = [m for m in move_analyses if m.is_problem]

        if with_comments and self._llm and problem_moves:
            comment_targets = [
                m for m in problem_moves
                if m.winrate_drop >= self._cfg.winrate_threshold
            ]
            if comment_targets:
                await self._generate_comments(game, comment_targets)

        curve = [initial_black_wr] + [m.black_winrate for m in move_analyses]

        result = ReviewResult(
            game_info=_build_game_info(game),
            move_analyses=move_analyses,
            problem_moves=problem_moves,
            black_winrate_curve=curve,
            initial_black_winrate=initial_black_wr,
        )

        n_major = sum(1 for m in problem_moves if m.severity in (Severity.QUESTIONABLE, Severity.BAD))
        logger.info(
            "复盘完成: %d 手, %d 个问题手 (%d 个疑问手/恶手)",
            result.total_moves, result.total_problems, n_major,
        )
        return result

    # ── KataGo 逐手分析 ──────────────────────────────────────

    async def _analyze_all_positions(
        self,
        game: ParsedGame,
        progress_callback: ProgressCallback = None,
    ) -> List[Dict[str, Any]]:
        """分析从初始局面到最终局面的所有位置（共 total_moves + 1 个）。

        每个位置对应 ``game.moves_up_to(i)``，其中 i ∈ [0, total_moves]。
        请求中包含 ``includeOwnership: true`` 以获取领地归属数据。
        """
        total = game.total_moves + 1
        results: List[Dict[str, Any]] = []

        for i in range(total):
            moves = game.moves_up_to(i)
            resp = await self._engine.query(
                moves=moves,
                rules=game.info.rules,
                komi=game.info.komi,
                board_x_size=game.info.board_size,
                board_y_size=game.info.board_size,
                extra={"includeOwnership": True},
            )
            results.append(resp)

            if progress_callback:
                await progress_callback(i + 1, total)

        return results

    # ── 胜率落差 & 问题手识别 ────────────────────────────────

    def _compute_move_analyses(
        self,
        game: ParsedGame,
        raw: List[Dict[str, Any]],
    ) -> List[MoveAnalysis]:
        """对比相邻位置的 KataGo 评估值，计算每手的胜率落差并分级。

        KataGo rootInfo.winrate 是从「当前行棋方」视角给出的胜率。
        落子前 raw[i] 的 rootInfo 对应行棋方（即 move.color），
        落子后 raw[i+1] 的 rootInfo 对应对手方。
        因此行棋方落子后的实际胜率 = 1 − raw[i+1].rootInfo.winrate。
        """
        analyses: List[MoveAnalysis] = []
        bs = game.info.board_size

        for i, move in enumerate(game.moves):
            root_before = raw[i].get("rootInfo", {})
            root_after = raw[i + 1].get("rootInfo", {})

            wr_before = root_before.get("winrate", 0.5)
            score_before = root_before.get("scoreLead", 0.0)

            wr_after_opp = root_after.get("winrate", 0.5)
            score_after_opp = root_after.get("scoreLead", 0.0)

            wr_after = 1.0 - wr_after_opp
            score_after = -score_after_opp

            wr_drop = wr_before - wr_after
            score_drop = score_before - score_after

            black_wr = wr_after if move.color == "B" else wr_after_opp

            variations = self._extract_variations(raw[i])
            severity = _classify_severity(wr_drop)

            # — ownership 分析 —
            own_before_raw = raw[i].get("ownership")
            own_after_raw = raw[i + 1].get("ownership")

            territory_before = territory_after = None
            ownership_ctx = ""

            if own_before_raw and own_after_raw:
                own_bk_before = _to_black_perspective(own_before_raw, move.color == "B")
                own_bk_after = _to_black_perspective(own_after_raw, move.color != "B")

                territory_before = _estimate_territory(own_bk_before)
                territory_after = _estimate_territory(own_bk_after)

                if severity is not None:
                    ownership_ctx = _build_ownership_context(
                        own_bk_before, own_bk_after,
                        territory_before, territory_after,
                        move.color, bs,
                    )

            analyses.append(MoveAnalysis(
                move_number=move.move_number,
                color=move.color,
                gtp_coord=move.gtp_coord,
                winrate_before=wr_before,
                winrate_after=wr_after,
                winrate_drop=wr_drop,
                score_before=score_before,
                score_after=score_after,
                score_drop=score_drop,
                black_winrate=black_wr,
                best_variations=variations,
                is_problem=severity is not None,
                severity=severity,
                territory_before=territory_before,
                territory_after=territory_after,
                ownership_context=ownership_ctx,
            ))

        return analyses

    # ── 变化提取 ──────────────────────────────────────────────

    def _extract_variations(self, resp: Dict[str, Any]) -> List[Variation]:
        """从 KataGo 响应中提取前 N 个候选变化（含后续手数限制）。"""
        infos = resp.get("moveInfos", [])
        top_n = self._cfg.top_variations
        depth = self._cfg.variation_depth

        return [
            Variation(
                move=info.get("move", "?"),
                winrate=info.get("winrate", 0.0),
                score_lead=info.get("scoreLead", 0.0),
                visits=info.get("visits", 0),
                pv=info.get("pv", [])[:depth],
            )
            for info in infos[:top_n]
        ]

    # ── LLM 解说 ──────────────────────────────────────────────

    async def _generate_comments(
        self,
        game: ParsedGame,
        targets: List[MoveAnalysis],
    ) -> None:
        """为目标问题手逐个生成 LLM 解说，结果直接写入 MoveAnalysis.comment。"""
        logger.info("为 %d 个问题手生成 LLM 解说 …", len(targets))
        for idx, ma in enumerate(targets):
            prompt = self._build_comment_prompt(game, ma)
            try:
                ma.comment = (await self._llm.chat(prompt)).strip()
                logger.debug("第 %d 手解说完成 (%d/%d)", ma.move_number, idx + 1, len(targets))
            except Exception:
                logger.exception("LLM 解说失败: 第 %d 手", ma.move_number)
                ma.comment = ""

    def _build_comment_prompt(self, game: ParsedGame, ma: MoveAnalysis) -> str:
        """按「目数 → 厚薄 → 死活」优先级为问题手组装 LLM prompt。"""
        sev_text = _SEVERITY_CN.get(ma.severity, "问题手")
        color_cn = "黑" if ma.color == "B" else "白"

        # — 候选变化 —
        var_lines: List[str] = []
        for j, v in enumerate(ma.best_variations, 1):
            pv_str = " → ".join(v.pv[:6]) if v.pv else "无"
            var_lines.append(
                f"  {j}. {v.move}  胜率={v.winrate:.1%}  目差={v.score_lead:+.1f}  "
                f"访问={v.visits}  变化={pv_str}"
            )
        vars_text = "\n".join(var_lines) or "  无推荐变化"

        # — ownership 分析段落 —
        ownership_section = ""
        if ma.ownership_context:
            ownership_section = (
                f"\n{ma.ownership_context}\n"
            )

        return (
            f"棋谱: {game.info.black_player}(黑) vs {game.info.white_player}(白)\n"
            f"规则: {game.info.rules}  贴目: {game.info.komi}\n\n"
            f"以下是 KataGo 对{color_cn}棋第 {ma.move_number} 手 "
            f"{ma.gtp_coord} 的分析:\n"
            f"- 判定: {sev_text}\n"
            f"- 胜率变化: 从 {ma.winrate_before:.1%} 跌至 {ma.winrate_after:.1%} "
            f"(跌幅 {ma.winrate_drop:.1%})\n"
            f"- 目差变化: 从 {ma.score_before:+.1f} 变为 {ma.score_after:+.1f} "
            f"(落差 {ma.score_drop:+.1f})\n"
            f"- KataGo 推荐着手:\n{vars_text}\n"
            f"{ownership_section}\n"
            f"请严格按照以下优先级分析这步棋:\n"
            f"1. 【目数/实地】(最高优先级): 这步棋导致了多少目的实地得失？"
            f"哪个区域的实地受到了影响？与推荐手相比损失了多少目？\n"
            f"2. 【厚薄/势力】(第二优先级): 这步棋是否让棋形变薄了？"
            f"是否放弃了重要的外势或发展潜力？是否留下了后续的薄味？\n"
            f"3. 【死活】(如果涉及): 这步棋是否危及了某个棋群的安全？"
            f"是否错过了攻击对方薄弱棋群的时机？\n\n"
            f"请用 2-4 段话回答。先讲清楚目数上亏了多少、为什么亏，"
            f"再从厚薄角度补充战略层面的影响，最后如果涉及死活也要提到。"
            f"使用围棋术语但照顾业余棋手理解水平。"
        )


# ── Ownership 分析工具函数 ────────────────────────────────────


def _to_black_perspective(
    ownership: List[float],
    is_black_turn: bool,
) -> List[float]:
    """将 KataGo ownership 数组统一转换为黑方视角。

    KataGo 的 ownership 是从「当前行棋方」视角给出的，
    正值=行棋方领地，负值=对手领地。
    统一转为正值=黑方领地后，全局数据才能直接对比。
    """
    if is_black_turn:
        return list(ownership)
    return [-v for v in ownership]


def _estimate_territory(ownership_black: List[float]) -> TerritoryEstimate:
    """从黑方视角的 ownership 估算双方目数。

    每个交叉点的 ownership 绝对值可以视为该点属于某方的概率，
    加权求和即得到期望目数。
    """
    black = sum(v for v in ownership_black if v > 0)
    white = sum(-v for v in ownership_black if v < 0)
    return TerritoryEstimate(black_territory=black, white_territory=white)


def _get_region_stats(
    ownership: List[float],
    board_size: int,
) -> Dict[str, float]:
    """计算每个棋盘区域的平均 ownership（黑方视角）。"""
    if board_size != 19:
        return {}
    stats: Dict[str, float] = {}
    for name, (rows, cols) in _REGIONS_19.items():
        vals = [ownership[y * board_size + x] for y in rows for x in cols]
        stats[name] = sum(vals) / len(vals) if vals else 0.0
    return stats


def _build_ownership_context(
    own_before: List[float],
    own_after: List[float],
    terr_before: TerritoryEstimate,
    terr_after: TerritoryEstimate,
    move_color: str,
    board_size: int,
) -> str:
    """从 ownership 前后对比生成「目数 / 厚薄 / 死活」三段式分析文本。"""
    lines: List[str] = []
    color_cn = "黑" if move_color == "B" else "白"
    opp_cn = "白" if move_color == "B" else "黑"

    # ── 1. 目数/实地分析 ──────────────────────────────────────
    lines.append("【目数/实地分析】")
    lines.append(
        f"- 落子前预估: 黑方约 {terr_before.black_territory:.1f} 目, "
        f"白方约 {terr_before.white_territory:.1f} 目 "
        f"(黑方{'领先' if terr_before.diff >= 0 else '落后'} {abs(terr_before.diff):.1f} 目)"
    )
    lines.append(
        f"- 落子后预估: 黑方约 {terr_after.black_territory:.1f} 目, "
        f"白方约 {terr_after.white_territory:.1f} 目 "
        f"(黑方{'领先' if terr_after.diff >= 0 else '落后'} {abs(terr_after.diff):.1f} 目)"
    )

    terr_change = terr_after.diff - terr_before.diff
    if move_color == "B":
        self_loss = -terr_change if terr_change < 0 else 0
        desc = f"黑方实地净损失约 {self_loss:.1f} 目" if self_loss > 0.5 else "黑方实地变化不大"
    else:
        self_loss = terr_change if terr_change > 0 else 0
        desc = f"白方实地净损失约 {self_loss:.1f} 目" if self_loss > 0.5 else "白方实地变化不大"
    lines.append(f"- {desc}")

    # ── 2. 厚薄/势力分析 ──────────────────────────────────────
    lines.append("")
    lines.append("【厚薄/势力分析】")

    if board_size == 19:
        region_before = _get_region_stats(own_before, board_size)
        region_after = _get_region_stats(own_after, board_size)

        changes: List[Tuple[str, float, float, float]] = []
        for name in region_before:
            avg_b = region_before[name]
            avg_a = region_after[name]
            delta = avg_a - avg_b
            if abs(delta) >= _REGION_CHANGE_MIN:
                changes.append((name, avg_b, avg_a, delta))

        changes.sort(key=lambda x: abs(x[3]), reverse=True)

        if changes:
            for name, avg_b, avg_a, delta in changes[:4]:
                owner_b = _describe_ownership(avg_b)
                owner_a = _describe_ownership(avg_a)
                direction = "黑方增强" if delta > 0 else "白方增强"
                lines.append(
                    f"- {name}: {owner_b} → {owner_a} ({direction} {abs(delta):.2f})"
                )
        else:
            lines.append("- 各区域势力分布变化不大")

        # 厚薄评估：统计势力区（0.2-0.8）的变化
        influence_before = sum(1 for v in own_before if _OWN_INFLUENCE < abs(v) < _OWN_LIKELY)
        influence_after = sum(1 for v in own_after if _OWN_INFLUENCE < abs(v) < _OWN_LIKELY)
        uncertain_before = sum(1 for v in own_before if abs(v) <= _OWN_INFLUENCE)
        uncertain_after = sum(1 for v in own_after if abs(v) <= _OWN_INFLUENCE)

        if uncertain_after > uncertain_before + 3:
            lines.append(f"- 注意: 不确定区域增加了 {uncertain_after - uncertain_before} 个点, "
                         f"{color_cn}棋局面可能变薄")
        elif influence_after < influence_before - 3:
            lines.append(f"- {color_cn}方势力区缩小, 外势潜力有所下降")
    else:
        lines.append("- (非 19 路棋盘, 区域分析省略)")

    # ── 3. 死活信号 ───────────────────────────────────────────
    life_death_alerts = _detect_life_death_signals(own_before, own_after, board_size)
    if life_death_alerts:
        lines.append("")
        lines.append("【死活信号】")
        for alert in life_death_alerts:
            lines.append(f"- {alert}")

    return "\n".join(lines)


def _describe_ownership(avg: float) -> str:
    """将区域平均 ownership 转化为中文描述。"""
    if avg > _OWN_CONFIRMED:
        return "黑方确定地盘"
    elif avg > _OWN_LIKELY:
        return "黑方优势区域"
    elif avg > _OWN_INFLUENCE:
        return "黑方薄势力"
    elif avg > -_OWN_INFLUENCE:
        return "中立/争夺区"
    elif avg > -_OWN_LIKELY:
        return "白方薄势力"
    elif avg > -_OWN_CONFIRMED:
        return "白方优势区域"
    else:
        return "白方确定地盘"


def _detect_life_death_signals(
    own_before: List[float],
    own_after: List[float],
    board_size: int,
) -> List[str]:
    """检测 ownership 骤变点，提示可能的死活相关变化。

    若某个区域内多个交叉点的归属从一方急剧翻转到另一方，
    说明可能有棋群被吃或者活了。
    """
    if board_size != 19:
        return []

    flip_black_to_white: Dict[str, int] = {}
    flip_white_to_black: Dict[str, int] = {}

    for y in range(board_size):
        for x in range(board_size):
            idx = y * board_size + x
            before = own_before[idx]
            after = own_after[idx]

            if before > _OWN_INFLUENCE and after < -_OWN_INFLUENCE:
                flip = before - after
                if flip >= _LIFE_DEATH_FLIP:
                    region = _point_to_region(y, x)
                    flip_black_to_white[region] = flip_black_to_white.get(region, 0) + 1

            elif before < -_OWN_INFLUENCE and after > _OWN_INFLUENCE:
                flip = after - before
                if flip >= _LIFE_DEATH_FLIP:
                    region = _point_to_region(y, x)
                    flip_white_to_black[region] = flip_white_to_black.get(region, 0) + 1

    alerts: List[str] = []
    for region, count in sorted(flip_black_to_white.items(), key=lambda x: -x[1]):
        if count >= 3:
            alerts.append(f"{region}有 {count} 个点从黑方归属骤变为白方, 黑棋可能存在死活危机")
    for region, count in sorted(flip_white_to_black.items(), key=lambda x: -x[1]):
        if count >= 3:
            alerts.append(f"{region}有 {count} 个点从白方归属骤变为黑方, 白棋可能存在死活危机")

    return alerts


def _point_to_region(y: int, x: int) -> str:
    """将棋盘坐标归入九宫格区域名称。"""
    for name, (rows, cols) in _REGIONS_19.items():
        if y in rows and x in cols:
            return name
    return "中腹"


# ── 其他工具函数 ──────────────────────────────────────────────


def _build_game_info(game: ParsedGame) -> Dict[str, Any]:
    info = game.info
    return {
        "blackPlayer": info.black_player,
        "whitePlayer": info.white_player,
        "blackRank": info.black_rank,
        "whiteRank": info.white_rank,
        "result": info.result,
        "komi": info.komi,
        "boardSize": info.board_size,
        "handicap": info.handicap,
        "rules": info.rules,
        "date": info.date,
        "event": info.event,
        "gameName": info.game_name,
    }


# ── 独立测试入口 ──────────────────────────────────────────────


async def _test():
    """端到端测试: 解析示例 SGF → KataGo 分析 → 打印问题手。"""
    from config import load_config
    from sgf_parser import parse_sgf

    cfg = load_config()
    engine = KataGoEngine(cfg.katago)
    llm = LLMService(cfg.llm)
    reviewer = Reviewer(engine, cfg.review, llm)

    sample_sgf = (
        "(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[7.5]"
        "PB[黑方棋手]PW[白方棋手]BR[9d]WR[9d]RE[B+R]"
        "RU[Chinese]DT[2025-01-01]"
        ";B[pd];W[dd];B[pq];W[dp];B[fq];W[cn]"
        ";B[qo];W[jp];B[cq];W[dq])"
    )
    game = parse_sgf(sample_sgf)

    async def on_progress(cur: int, total: int):
        print(f"\r分析进度: {cur}/{total}", end="", flush=True)

    await engine.start()
    try:
        result = await reviewer.review_game(
            game, with_comments=False, progress_callback=on_progress,
        )
    finally:
        await engine.stop()

    print(f"\n\n{'='*60}")
    print(f"棋谱: {game.info.black_player} vs {game.info.white_player}")
    print(f"总手数: {result.total_moves}  问题手: {result.total_problems}")
    print(f"初始黑方胜率: {result.initial_black_winrate:.1%}")
    print(f"{'='*60}\n")

    print("胜率曲线 (黑方视角):")
    for i, wr in enumerate(result.black_winrate_curve):
        bar = "█" * int(wr * 40)
        label = "初始" if i == 0 else f"第{i:>3}手"
        print(f"  {label} | {bar} {wr:.1%}")

    print(f"\n问题手列表:")
    for ma in result.problem_moves:
        sev = _SEVERITY_CN.get(ma.severity, "?")
        color = "黑" if ma.color == "B" else "白"
        print(
            f"  第 {ma.move_number:>3} 手  {color} {ma.gtp_coord:<4}  "
            f"{sev}  跌幅={ma.winrate_drop:.1%}"
        )
        if ma.best_variations:
            best = ma.best_variations[0]
            print(f"         推荐: {best.move} (胜率 {best.winrate:.1%})")
        if ma.territory_before and ma.territory_after:
            print(f"         目数: 黑{ma.territory_before.black_territory:.1f} "
                  f"白{ma.territory_before.white_territory:.1f} → "
                  f"黑{ma.territory_after.black_territory:.1f} "
                  f"白{ma.territory_after.white_territory:.1f}")
        if ma.ownership_context:
            print(f"         --- ownership 分析 ---")
            for line in ma.ownership_context.split("\n"):
                print(f"         {line}")

    print(f"\n完整 JSON 片段 (第一个问题手):")
    if result.problem_moves:
        import json
        print(json.dumps(result.problem_moves[0].to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(_test())
