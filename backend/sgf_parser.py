"""SGF 解析模块 — 基于 sgfmill 提取棋谱数据

用法:
    game = parse_sgf(sgf_text)
    print(game.info.black_player, "vs", game.info.white_player)
    print(f"共 {game.total_moves} 手")
    katago_moves = game.moves_up_to(42)  # 前 42 手的 KataGo 格式列表
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sgfmill import sgf

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"


# ── 数据结构 ──────────────────────────────────────────────────


@dataclass
class GameInfo:
    """棋谱元信息。"""

    black_player: str = "Unknown"
    white_player: str = "Unknown"
    black_rank: str = ""
    white_rank: str = ""
    result: str = ""
    komi: float = 7.5
    board_size: int = 19
    handicap: int = 0
    rules: str = "chinese"
    date: str = ""
    event: str = ""
    game_name: str = ""


@dataclass
class Move:
    """单步着手。"""

    move_number: int
    color: str  # "B" / "W"
    gtp_coord: str  # GTP 坐标如 "D4"；虚手为 "pass"
    row: Optional[int] = None  # sgfmill 行号 0-indexed（0 = 底行）
    col: Optional[int] = None  # sgfmill 列号 0-indexed（0 = 左列）

    @property
    def is_pass(self) -> bool:
        return self.gtp_coord == "pass"

    def to_katago(self) -> List[str]:
        """转为 KataGo analysis 所需的 ``[color, coord]`` 格式。"""
        return [self.color, self.gtp_coord]


@dataclass
class ParsedGame:
    """解析后的完整棋谱。"""

    info: GameInfo
    moves: List[Move]
    setup_stones: List[List[str]] = field(default_factory=list)

    @property
    def total_moves(self) -> int:
        return len(self.moves)

    def moves_up_to(self, n: int) -> List[List[str]]:
        """返回前 *n* 手的 KataGo 格式着手列表（含让子摆放）。

        每个元素为 ``["B", "D4"]`` 或 ``["W", "Q16"]``。
        """
        result: List[List[str]] = [s[:] for s in self.setup_stones]
        for m in self.moves[:n]:
            result.append(m.to_katago())
        return result

    def katago_moves(self) -> List[List[str]]:
        """返回全部着手的 KataGo 格式列表。"""
        return self.moves_up_to(self.total_moves)


# ── 坐标转换 ──────────────────────────────────────────────────


def _point_to_gtp(row: int, col: int, board_size: int = 19) -> str:
    """sgfmill ``(row, col)`` → GTP 坐标字符串。

    sgfmill: row 0 = 棋盘底行, col 0 = 最左列
    GTP:     列 A-T（跳过 I），行 1-19（1 = 底行）
    """
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"坐标越界: row={row}, col={col}, board_size={board_size}")
    return f"{_GTP_COLS[col]}{row + 1}"


def gtp_to_point(gtp: str, board_size: int = 19) -> Tuple[int, int]:
    """GTP 坐标字符串 → sgfmill ``(row, col)``。"""
    if not gtp or gtp.lower() == "pass":
        raise ValueError("pass 没有对应棋盘坐标")
    letter = gtp[0].upper()
    col = _GTP_COLS.index(letter)
    row = int(gtp[1:]) - 1
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"GTP 坐标越界: {gtp}, board_size={board_size}")
    return row, col


# ── 解析函数 ──────────────────────────────────────────────────


def parse_sgf(sgf_text: str) -> ParsedGame:
    """解析 SGF 字符串，返回 :class:`ParsedGame`。

    许多围棋服务器（Fox Weiqi、弈城等）导出的 SGF 使用 UTF-8 编码
    但不包含 ``CA[UTF-8]`` 声明，导致 sgfmill 按 latin-1 解码产生乱码。
    本函数会自动检测并注入编码声明。

    Raises:
        ValueError: SGF 格式无效。
    """
    if isinstance(sgf_text, str):
        sgf_bytes = sgf_text.encode("utf-8")
    else:
        sgf_bytes = sgf_text

    if b"CA[" not in sgf_bytes and b"(;" in sgf_bytes:
        sgf_bytes = sgf_bytes.replace(b"(;", b"(;CA[UTF-8]", 1)

    try:
        game = sgf.Sgf_game.from_bytes(sgf_bytes)
    except ValueError as exc:
        raise ValueError(f"SGF 格式无效: {exc}") from exc

    root = game.get_root()
    board_size = game.get_size()

    info = _extract_info(root, board_size)
    setup_stones = _extract_setup_stones(root, board_size)
    moves = _extract_moves(game, board_size)

    return ParsedGame(info=info, moves=moves, setup_stones=setup_stones)


def parse_sgf_file(path: Union[str, Path]) -> ParsedGame:
    """从文件路径解析 SGF。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SGF 文件不存在: {p}")
    raw = p.read_bytes()
    return parse_sgf(raw.decode("utf-8", errors="replace"))


# ── 内部辅助 ──────────────────────────────────────────────────


def _sgf_get(node, prop: str, default=None):
    """安全读取 SGF 节点属性，缺失时返回 *default*。"""
    try:
        return node.get(prop)
    except KeyError:
        return default


def _extract_info(root, board_size: int) -> GameInfo:
    """从根节点提取元信息。"""
    komi_raw = _sgf_get(root, "KM")
    try:
        komi = float(komi_raw)
    except (ValueError, TypeError):
        komi = 7.5
    if abs(komi) > 100:
        komi = komi / 100.0 if abs(komi) > 300 else komi / 10.0

    handicap_raw = _sgf_get(root, "HA")
    try:
        handicap = int(handicap_raw)
    except (ValueError, TypeError):
        handicap = 0

    return GameInfo(
        black_player=_sgf_get(root, "PB", "Unknown") or "Unknown",
        white_player=_sgf_get(root, "PW", "Unknown") or "Unknown",
        black_rank=_sgf_get(root, "BR", "") or "",
        white_rank=_sgf_get(root, "WR", "") or "",
        result=_sgf_get(root, "RE", "") or "",
        komi=komi,
        board_size=board_size,
        handicap=handicap,
        rules=_sgf_get(root, "RU", "chinese") or "chinese",
        date=_sgf_get(root, "DT", "") or "",
        event=_sgf_get(root, "EV", "") or "",
        game_name=_sgf_get(root, "GN", "") or "",
    )


def _extract_setup_stones(root, board_size: int) -> List[List[str]]:
    """提取让子（AB/AW）摆放石。"""
    stones: List[List[str]] = []
    if not root.has_setup_stones():
        return stones
    black_pts, white_pts, _ = root.get_setup_stones()
    for r, c in sorted(black_pts):
        stones.append(["B", _point_to_gtp(r, c, board_size)])
    for r, c in sorted(white_pts):
        stones.append(["W", _point_to_gtp(r, c, board_size)])
    return stones


def _extract_moves(game, board_size: int) -> List[Move]:
    """遍历主线序列，提取每步着手。"""
    moves: List[Move] = []
    move_num = 0
    for node in game.get_main_sequence():
        color_raw, point = node.get_move()
        if color_raw is None:
            continue
        move_num += 1
        color = color_raw.upper()

        if point is None:
            moves.append(Move(move_number=move_num, color=color, gtp_coord="pass"))
        else:
            row, col = point
            moves.append(Move(
                move_number=move_num,
                color=color,
                gtp_coord=_point_to_gtp(row, col, board_size),
                row=row,
                col=col,
            ))
    return moves


# ── 独立测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    sample_sgf = (
        "(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[7.5]"
        "PB[黑方棋手]PW[白方棋手]BR[9d]WR[9d]RE[B+R]"
        "RU[Chinese]DT[2025-01-01]"
        ";B[pd];W[dd];B[pq];W[dp];B[fq];W[cn]"
        ";B[qo];W[jp];B[cq];W[dq])"
    )
    parsed = parse_sgf(sample_sgf)
    print(f"黑方: {parsed.info.black_player} ({parsed.info.black_rank})")
    print(f"白方: {parsed.info.white_player} ({parsed.info.white_rank})")
    print(f"结果: {parsed.info.result}")
    print(f"贴目: {parsed.info.komi}")
    print(f"规则: {parsed.info.rules}")
    print(f"总手数: {parsed.total_moves}")
    print()
    for m in parsed.moves:
        tag = "（虚手）" if m.is_pass else ""
        print(f"  第 {m.move_number:>3} 手  {m.color}  {m.gtp_coord:<4} {tag}")
    print()
    print("KataGo 格式（前 5 手）:", parsed.moves_up_to(5))
