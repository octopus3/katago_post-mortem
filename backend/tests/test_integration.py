"""端到端集成测试 — 用实际棋谱验证完整复盘流程

测试覆盖:
    1. SGF 解析 → 棋谱元信息、着手序列、坐标转换
    2. Reviewer 核心逻辑 → 胜率落差、问题手分级、目数估算 (mock KataGo)
    3. LLM 解说生成 → prompt 构建与注入 (mock LLM)
    4. FastAPI 端点 → 上传 / 分析 / 轮询 / 查询全流程
    5. 序列化 → ReviewResult.to_dict() 的 JSON 可序列化性
    6. 真实对局棋谱 → Fox Weiqi SGF 解析 + 全局复盘流程
"""

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import httpx
import pytest

from config import ReviewConfig
from reviewer import MoveAnalysis, ReviewResult, Reviewer, Severity
from sgf_parser import ParsedGame, parse_sgf

# ── 测试棋谱（10 手标准开局） ─────────────────────────────────

SAMPLE_SGF = (
    "(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[7.5]"
    "PB[AlphaGo]PW[Lee Sedol]BR[9d]WR[9p]RE[B+R]"
    "RU[Chinese]DT[2016-03-09]EV[Test Match]GN[Game 1]"
    ";B[pd];W[dp];B[cd];W[qp];B[op];W[oq]"
    ";B[nq];W[pq];B[cn];W[fq])"
)

EXPECTED_MOVES_GTP = [
    ("B", "Q16"),   # 1. pd
    ("W", "D4"),    # 2. dp
    ("B", "C16"),   # 3. cd
    ("W", "R4"),    # 4. qp
    ("B", "P4"),    # 5. op
    ("W", "P3"),    # 6. oq
    ("B", "O3"),    # 7. nq
    ("W", "Q3"),    # 8. pq
    ("B", "C6"),    # 9. cn
    ("W", "F3"),    # 10. fq
]

# ── KataGo Mock 数据 ──────────────────────────────────────────
#
# rootInfo.winrate 从「当前行棋方」视角给出。
# 设计要点:
#   Move 5 (B P4)  → wr_drop ≈ 0.07 → Severity.QUESTIONABLE
#   Move 7 (B O3)  → wr_drop ≈ 0.14 → Severity.BAD
#   Move 10 (W F3) → wr_drop ≈ 0.02 → Severity.MINOR

_MOCK_WINRATES = [
    0.50,  # pos  0: B to move (empty board)
    0.48,  # pos  1: W to move
    0.51,  # pos  2: B to move
    0.47,  # pos  3: W to move
    0.54,  # pos  4: B to move
    0.53,  # pos  5: W to move (after B bad move 5)
    0.46,  # pos  6: B to move
    0.68,  # pos  7: W to move (after B terrible move 7)
    0.33,  # pos  8: B to move
    0.65,  # pos  9: W to move
    0.37,  # pos 10: B to move (after W minor mistake 10)
]

_MOCK_SCORES = [
    0.0, -0.5, 0.3, -0.8, 1.5,
    1.0, -1.0, 5.0, -5.5, 4.0, -3.5,
]

_CANDIDATE_COORDS = ["D4", "Q16", "R14", "C6", "O3"]


def _make_ownership(n_moves: int) -> List[float]:
    """生成 361 个 mock ownership 值（黑正白负）。"""
    result = []
    for r in range(19):
        for c in range(19):
            if r < 6 and c > 12:
                val = 0.4 + 0.01 * n_moves
            elif r > 12 and c < 6:
                val = -0.4 - 0.01 * n_moves
            elif r < 6 and c < 6:
                val = 0.1
            elif r > 12 and c > 12:
                val = -0.1
            else:
                val = 0.0
            result.append(max(-1.0, min(1.0, val)))
    return result


def _make_move_infos(n_moves: int) -> List[Dict[str, Any]]:
    wr = _MOCK_WINRATES[n_moves] if n_moves < len(_MOCK_WINRATES) else 0.5
    sc = _MOCK_SCORES[n_moves] if n_moves < len(_MOCK_SCORES) else 0.0
    infos = []
    for j in range(3):
        move = _CANDIDATE_COORDS[(n_moves + j) % len(_CANDIDATE_COORDS)]
        infos.append({
            "move": move,
            "winrate": wr + 0.01 * (2 - j),
            "scoreLead": sc,
            "visits": 500 - j * 100,
            "pv": [
                _CANDIDATE_COORDS[(n_moves + j + k) % len(_CANDIDATE_COORDS)]
                for k in range(5)
            ],
        })
    return infos


def _make_katago_response(query_id: str, n_moves: int) -> Dict[str, Any]:
    wr = _MOCK_WINRATES[n_moves] if n_moves < len(_MOCK_WINRATES) else 0.5
    sc = _MOCK_SCORES[n_moves] if n_moves < len(_MOCK_SCORES) else 0.0
    return {
        "id": query_id,
        "rootInfo": {"winrate": wr, "scoreLead": sc, "visits": 500},
        "moveInfos": _make_move_infos(n_moves),
        "ownership": _make_ownership(n_moves),
    }


# ── Mock 类 ──────────────────────────────────────────────────


class MockKataGoEngine:
    """模拟 KataGo 引擎，根据着手数量返回预设分析数据。"""

    is_running = True

    def __init__(self):
        self._counter = 0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def query(
        self,
        moves,
        rules="chinese",
        komi=7.5,
        max_visits=None,
        board_x_size=19,
        board_y_size=19,
        extra=None,
    ):
        self._counter += 1
        return _make_katago_response(f"q_{self._counter}", len(moves))


class MockLLMService:
    """模拟 LLM 服务，返回固定的围棋解说文本。"""

    call_count = 0

    async def chat(
        self,
        user_message,
        system_prompt=None,
        history=None,
        temperature=0.7,
        max_tokens=1024,
    ):
        self.call_count += 1
        return (
            "这步棋在目数上有明显损失。实战下在这里，"
            "让对方在右下角获得了约3目的实地便宜。"
            "推荐的下法是在上边展开，兼顾实地和外势。"
        )


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def game():
    return parse_sgf(SAMPLE_SGF)


@pytest.fixture
def mock_engine():
    return MockKataGoEngine()


@pytest.fixture
def mock_llm():
    return MockLLMService()


@pytest.fixture
def review_cfg():
    return ReviewConfig(winrate_threshold=0.05, top_variations=3, variation_depth=8)


# ══════════════════════════════════════════════════════════════
# 第一部分: SGF 解析
# ══════════════════════════════════════════════════════════════


class TestSGFParsing:

    def test_game_info_fields(self, game: ParsedGame):
        info = game.info
        assert info.black_player == "AlphaGo"
        assert info.white_player == "Lee Sedol"
        assert info.black_rank == "9d"
        assert info.white_rank == "9p"
        assert info.result == "B+R"
        assert info.komi == 7.5
        assert info.board_size == 19
        assert info.rules.lower() == "chinese"
        assert info.date == "2016-03-09"
        assert info.event == "Test Match"
        assert info.game_name == "Game 1"

    def test_total_moves(self, game: ParsedGame):
        assert game.total_moves == 10

    def test_move_coordinates(self, game: ParsedGame):
        for i, (expected_color, expected_gtp) in enumerate(EXPECTED_MOVES_GTP):
            m = game.moves[i]
            assert m.color == expected_color, f"第{i+1}手颜色不匹配"
            assert m.gtp_coord == expected_gtp, f"第{i+1}手坐标不匹配"
            assert m.move_number == i + 1

    def test_no_pass_moves(self, game: ParsedGame):
        for m in game.moves:
            assert not m.is_pass

    def test_moves_up_to(self, game: ParsedGame):
        first_3 = game.moves_up_to(3)
        assert len(first_3) == 3
        assert first_3[0] == ["B", "Q16"]
        assert first_3[1] == ["W", "D4"]
        assert first_3[2] == ["B", "C16"]

    def test_katago_moves_all(self, game: ParsedGame):
        all_moves = game.katago_moves()
        assert len(all_moves) == 10
        for m in all_moves:
            assert len(m) == 2
            assert m[0] in ("B", "W")

    def test_no_setup_stones(self, game: ParsedGame):
        assert game.setup_stones == []

    def test_invalid_sgf_raises(self):
        with pytest.raises(ValueError, match="SGF"):
            parse_sgf("not valid sgf at all")

    def test_empty_sgf_raises(self):
        with pytest.raises(ValueError):
            parse_sgf("")

    def test_parse_sgf_with_pass(self):
        sgf = (
            "(;GM[1]FF[4]SZ[19]KM[7.5]PB[B]PW[W]"
            ";B[pd];W[dp];B[];W[dd])"
        )
        game = parse_sgf(sgf)
        assert game.total_moves == 4
        assert game.moves[2].is_pass
        assert game.moves[2].gtp_coord == "pass"

    def test_auto_inject_ca_utf8(self):
        """缺少 CA[UTF-8] 的 SGF 应自动注入编码声明，中文正常解析。"""
        sgf_no_ca = "(;GM[1]FF[4]SZ[19]KM[7.5]PB[柯洁]PW[李世石];B[pd];W[dp])"
        game = parse_sgf(sgf_no_ca)
        assert game.info.black_player == "柯洁"
        assert game.info.white_player == "李世石"

    def test_existing_ca_not_duplicated(self):
        """已有 CA[UTF-8] 的 SGF 不应重复注入。"""
        sgf_with_ca = (
            "(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[7.5]PB[柯洁]PW[李世石]"
            ";B[pd];W[dp])"
        )
        game = parse_sgf(sgf_with_ca)
        assert game.info.black_player == "柯洁"
        assert game.info.white_player == "李世石"


# ══════════════════════════════════════════════════════════════
# 第二部分: Reviewer 核心逻辑 (mock KataGo)
# ══════════════════════════════════════════════════════════════


class TestReviewerPipeline:

    @pytest.mark.asyncio
    async def test_full_review_structure(self, mock_engine, review_cfg, game):
        """完整复盘流程（不含 LLM），验证结果结构。"""
        reviewer = Reviewer(mock_engine, review_cfg, llm=None)

        progress_calls = []

        async def on_progress(current, total):
            progress_calls.append((current, total))

        result = await reviewer.review_game(
            game, with_comments=False, progress_callback=on_progress,
        )

        assert isinstance(result, ReviewResult)
        assert result.total_moves == 10
        assert result.game_info["blackPlayer"] == "AlphaGo"
        assert result.game_info["whitePlayer"] == "Lee Sedol"

        # 11 个位置 → 11 次进度回调
        assert len(progress_calls) == 11
        assert progress_calls[0] == (1, 11)
        assert progress_calls[-1] == (11, 11)

    @pytest.mark.asyncio
    async def test_problem_moves_detection(self, mock_engine, review_cfg, game):
        """验证问题手被正确识别并分级。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        assert result.total_problems == 3
        severity_map = {m.move_number: m.severity for m in result.problem_moves}

        assert severity_map[5] == Severity.QUESTIONABLE   # drop ≈ 0.07
        assert severity_map[7] == Severity.BAD             # drop ≈ 0.14
        assert severity_map[10] == Severity.MINOR          # drop ≈ 0.02

    @pytest.mark.asyncio
    async def test_winrate_drops(self, mock_engine, review_cfg, game):
        """验证各手的胜率落差数值。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        ma5 = result.move_analyses[4]   # move 5
        ma7 = result.move_analyses[6]   # move 7
        ma10 = result.move_analyses[9]  # move 10

        assert ma5.winrate_drop == pytest.approx(0.07, abs=0.001)
        assert ma7.winrate_drop == pytest.approx(0.14, abs=0.001)
        assert ma10.winrate_drop == pytest.approx(0.02, abs=0.001)

        # 非问题手的落差应小于 2%
        for ma in result.move_analyses:
            if ma.move_number not in (5, 7, 10):
                assert ma.winrate_drop < 0.02

    @pytest.mark.asyncio
    async def test_winrate_curve(self, mock_engine, review_cfg, game):
        """验证胜率曲线（黑方视角）。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        curve = result.black_winrate_curve
        assert len(curve) == 11  # initial + 10 moves
        assert curve[0] == pytest.approx(0.50, abs=0.01)

        # Move 7 bad → black winrate 急跌
        assert curve[7] < 0.35

        # 曲线单调性不作严格要求，但最终应偏向白方
        assert curve[-1] < 0.50

    @pytest.mark.asyncio
    async def test_variations_extracted(self, mock_engine, review_cfg, game):
        """验证每手都提取了候选变化。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        for ma in result.move_analyses:
            assert len(ma.best_variations) == review_cfg.top_variations
            for v in ma.best_variations:
                assert v.move
                assert v.visits > 0
                assert len(v.pv) > 0

    @pytest.mark.asyncio
    async def test_territory_estimates(self, mock_engine, review_cfg, game):
        """验证目数估算存在且合理。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        for ma in result.move_analyses:
            assert ma.territory_before is not None
            assert ma.territory_after is not None
            assert ma.territory_before.black_territory >= 0
            assert ma.territory_before.white_territory >= 0
            assert ma.territory_after.black_territory >= 0
            assert ma.territory_after.white_territory >= 0

    @pytest.mark.asyncio
    async def test_ownership_context_on_problems(self, mock_engine, review_cfg, game):
        """问题手应附带 ownership 分析文本。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        for ma in result.problem_moves:
            assert ma.ownership_context, f"第{ma.move_number}手缺少 ownership 分析"
            assert "目数" in ma.ownership_context or "实地" in ma.ownership_context

    @pytest.mark.asyncio
    async def test_review_with_llm_comments(self, mock_engine, mock_llm, review_cfg, game):
        """含 LLM 解说的完整流程，只有阈值以上的问题手获得解说。"""
        reviewer = Reviewer(mock_engine, review_cfg, llm=mock_llm)
        result = await reviewer.review_game(game, with_comments=True)

        commented = [m for m in result.problem_moves if m.comment]
        # Move 5 (drop=0.07≥0.05) 和 Move 7 (drop=0.14≥0.05) 应有解说
        # Move 10 (drop=0.02<0.05) 不应有解说
        assert len(commented) == 2

        for ma in commented:
            assert ma.winrate_drop >= review_cfg.winrate_threshold
            assert len(ma.comment) > 10
            assert ma.move_number in (5, 7)

        # Move 10 不应有解说
        ma10 = next(m for m in result.problem_moves if m.move_number == 10)
        assert ma10.comment == ""

    @pytest.mark.asyncio
    async def test_result_serialization(self, mock_engine, review_cfg, game):
        """ReviewResult 必须能完整序列化为 JSON。"""
        reviewer = Reviewer(mock_engine, review_cfg)
        result = await reviewer.review_game(game, with_comments=False)

        d = result.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)

        assert parsed["totalMoves"] == 10
        assert parsed["totalProblems"] == 3
        assert len(parsed["moveAnalyses"]) == 10
        assert len(parsed["blackWinrateCurve"]) == 11

        required_fields = [
            "moveNumber", "color", "gtpCoord",
            "winrateBefore", "winrateAfter", "winrateDrop",
            "scoreBefore", "scoreAfter", "scoreDrop",
            "blackWinrate", "bestVariations", "isProblem", "severity",
        ]
        for ma_dict in parsed["moveAnalyses"]:
            for field in required_fields:
                assert field in ma_dict, f"缺少字段 {field}"

    @pytest.mark.asyncio
    async def test_no_comments_when_llm_none(self, mock_engine, review_cfg, game):
        """llm=None 时即使 with_comments=True 也不生成解说。"""
        reviewer = Reviewer(mock_engine, review_cfg, llm=None)
        result = await reviewer.review_game(game, with_comments=True)
        for ma in result.move_analyses:
            assert ma.comment == ""


# ══════════════════════════════════════════════════════════════
# 第三部分: FastAPI 端点集成测试
# ══════════════════════════════════════════════════════════════


class TestAPIIntegration:

    @pytest.fixture
    def _patch_services(self, mock_engine, mock_llm):
        """统一 patch engine 和 llm_service。"""
        with patch("main.engine", mock_engine), \
             patch("main.llm_service", mock_llm):
            yield

    @pytest.fixture
    def app(self, _patch_services):
        from main import app
        return app

    # ── 上传 ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_upload_sgf_text(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})
            assert resp.status_code == 200
            data = resp.json()
            assert data["totalMoves"] == 10
            assert data["gameInfo"]["blackPlayer"] == "AlphaGo"
            assert data["gameInfo"]["whitePlayer"] == "Lee Sedol"
            assert "gameId" in data

    @pytest.mark.asyncio
    async def test_upload_sgf_file(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/api/upload",
                files={"file": ("game.sgf", SAMPLE_SGF.encode(), "application/octet-stream")},
            )
            assert resp.status_code == 200
            assert resp.json()["totalMoves"] == 10

    @pytest.mark.asyncio
    async def test_upload_invalid_sgf(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post("/api/upload/text", json={"sgf": "invalid sgf"})
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_upload_non_sgf_file(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/api/upload",
                files={"file": ("readme.txt", b"hello", "text/plain")},
            )
            assert resp.status_code == 400

    # ── 健康检查 & 棋谱列表 ──────────────────────────────

    @pytest.mark.asyncio
    async def test_health(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["katagoRunning"] is True

    @pytest.mark.asyncio
    async def test_games_list(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})
            resp = await c.get("/api/games")
            assert resp.status_code == 200
            games = resp.json()
            assert len(games) >= 1
            assert games[0]["blackPlayer"] == "AlphaGo"
            assert games[0]["totalMoves"] == 10

    # ── 棋谱详情 ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_game_detail(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            upload = (await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})).json()
            gid = upload["gameId"]

            resp = await c.get(f"/api/game/{gid}")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["moves"]) == 10
            assert data["moves"][0]["color"] == "B"
            assert data["moves"][0]["gtpCoord"] == "Q16"

    @pytest.mark.asyncio
    async def test_nonexistent_game_404(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/api/game/nonexistent_id")
            assert resp.status_code == 404

    # ── 完整复盘流程 ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_full_review_flow(self, app):
        """上传 → 启动分析 → 轮询状态 → 获取结果 → 查询细节。"""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            # 1) 上传
            upload = (await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})).json()
            gid = upload["gameId"]

            # 2) 启动复盘（不含 LLM 解说，加快速度）
            resp = await c.post(f"/api/review/{gid}", json={"withComments": False})
            assert resp.status_code == 200

            # 3) 轮询直到完成
            for _ in range(100):
                await asyncio.sleep(0.05)
                status = (await c.get(f"/api/review/{gid}/status")).json()
                if status["status"] == "done":
                    break
                if status["status"] == "error":
                    pytest.fail(f"分析出错: {status.get('error')}")
            else:
                pytest.fail("分析超时未完成")

            # 4) 获取完整结果
            resp = await c.get(f"/api/review/{gid}/result")
            assert resp.status_code == 200
            result = resp.json()
            assert result["totalMoves"] == 10
            assert result["totalProblems"] >= 1
            assert len(result["moveAnalyses"]) == 10
            assert len(result["blackWinrateCurve"]) == 11

            # 5) 获取问题手列表
            resp = await c.get(f"/api/review/{gid}/problems")
            assert resp.status_code == 200
            problems = resp.json()
            assert problems["totalProblems"] >= 1
            problem_numbers = {p["moveNumber"] for p in problems["problems"]}
            assert 7 in problem_numbers  # 恶手

            # 6) 获取胜率曲线
            resp = await c.get(f"/api/review/{gid}/winrate")
            assert resp.status_code == 200
            winrate = resp.json()
            assert len(winrate["curve"]) == 11

            # 7) 获取单手分析
            resp = await c.get(f"/api/review/{gid}/move/7")
            assert resp.status_code == 200
            move7 = resp.json()
            assert move7["moveNumber"] == 7
            assert move7["color"] == "B"
            assert move7["isProblem"] is True
            assert move7["severity"] == "bad"

            # 8) 正常手不应是问题手
            resp = await c.get(f"/api/review/{gid}/move/1")
            assert resp.status_code == 200
            move1 = resp.json()
            assert move1["isProblem"] is False
            assert move1["severity"] is None

    # ── 错误场景 ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_result_before_review_400(self, app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            gid = (await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})).json()["gameId"]
            resp = await c.get(f"/api/review/{gid}/result")
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_move_out_of_range(self, app):
        """分析完成后请求超出范围的手数应返回 404。"""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            gid = (await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})).json()["gameId"]
            await c.post(f"/api/review/{gid}", json={"withComments": False})
            for _ in range(100):
                await asyncio.sleep(0.05)
                s = (await c.get(f"/api/review/{gid}/status")).json()
                if s["status"] in ("done", "error"):
                    break

            resp = await c.get(f"/api/review/{gid}/move/999")
            assert resp.status_code == 404

    # ── 进度推送 ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_progress_increments(self, app):
        """轮询状态接口应显示进度递增。"""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            gid = (await c.post("/api/upload/text", json={"sgf": SAMPLE_SGF})).json()["gameId"]
            await c.post(f"/api/review/{gid}", json={"withComments": False})

            seen_progress = []
            for _ in range(100):
                await asyncio.sleep(0.05)
                s = (await c.get(f"/api/review/{gid}/status")).json()
                seen_progress.append(s["progressCurrent"])
                if s["status"] == "done":
                    break

            # 最终进度应等于 total
            assert seen_progress[-1] == 11


# ══════════════════════════════════════════════════════════════
# 第四部分: 真实对局棋谱端到端测试
# ══════════════════════════════════════════════════════════════

_REAL_SGF_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "test"
    / "[知更不咕鸟]vs[朕肖]1771817350030052653.sgf"
)


class DynamicMockKataGoEngine:
    """支持任意长度棋谱的 mock 引擎。

    使用正弦波生成自然波动的胜率，并在特定位置注入
    较大跳变以模拟问题手（恶手 / 疑问手）。
    """

    is_running = True

    # 在这些位置注入胜率跳变（position index, 跳变幅度）
    # 跳变位置处和前一位置同时升高 → 产生 wr_drop
    _SPIKE_POSITIONS = {
        30: 0.06, 31: 0.06,    # → move 30 BAD
        80: 0.06, 81: 0.06,    # → move 80 BAD
        150: 0.05, 151: 0.05,  # → move 150 QUESTIONABLE
        220: 0.06, 221: 0.06,  # → move 220 BAD
    }

    def __init__(self):
        self._counter = 0

    async def start(self):
        pass

    async def stop(self):
        pass

    async def query(
        self, moves, rules="chinese", komi=7.5,
        max_visits=None, board_x_size=19, board_y_size=19, extra=None,
    ):
        self._counter += 1
        n = len(moves)
        wr = self._winrate(n)
        sc = (wr - 0.5) * 20
        return {
            "id": f"dq_{self._counter}",
            "rootInfo": {"winrate": wr, "scoreLead": sc, "visits": 500},
            "moveInfos": [
                {
                    "move": coord,
                    "winrate": wr + 0.01 * (1 - j),
                    "scoreLead": sc,
                    "visits": 500 - j * 100,
                    "pv": [coord, "Q3", "D16"],
                }
                for j, coord in enumerate(["D4", "Q16", "R14"])
            ],
            "ownership": _make_ownership(n),
        }

    @classmethod
    def _winrate(cls, n: int) -> float:
        base = 0.50 + 0.02 * math.sin(n * 0.5) + 0.01 * math.cos(n * 0.3)
        base += cls._SPIKE_POSITIONS.get(n, 0.0)
        return max(0.05, min(0.95, base))


@pytest.mark.skipif(not _REAL_SGF_PATH.exists(), reason="真实对局棋谱文件不存在")
class TestRealGame:
    """使用 [知更不咕鸟] vs [朕肖] 真实对局验证完整流程。"""

    @pytest.fixture
    def real_sgf_text(self):
        return _REAL_SGF_PATH.read_text(encoding="utf-8")

    @pytest.fixture
    def real_game(self, real_sgf_text):
        return parse_sgf(real_sgf_text)

    @pytest.fixture
    def dynamic_engine(self):
        return DynamicMockKataGoEngine()

    # ── SGF 解析 ─────────────────────────────────────────

    def test_parse_game_info(self, real_game):
        """Fox Weiqi SGF 的元信息应正确解析（含中文棋手名）。"""
        info = real_game.info
        assert info.black_player == "知更不咕鸟"
        assert info.white_player == "朕肖"
        assert info.black_rank == "4级"
        assert info.white_rank == "3段"
        assert info.result == "W+8.25"
        assert info.board_size == 19
        assert info.rules.lower() == "chinese"
        assert info.date == "2026-02-23"

    def test_komi_fox_format(self, real_game):
        """Fox Weiqi 的 KM[375] 按 SGF 标准被解析为 375.0。"""
        assert real_game.info.komi == 375.0

    def test_move_count(self, real_game):
        """整局棋应有 200+ 手。"""
        assert real_game.total_moves > 200

    def test_first_move_is_black(self, real_game):
        assert real_game.moves[0].color == "B"

    def test_opening_sequence(self, real_game):
        """验证开局几手坐标与 SGF 一致。"""
        expected = [
            ("B", "Q16"),  # pd
            ("W", "D4"),   # dp
            ("B", "Q3"),   # pq
            ("W", "D17"),  # dc
            ("B", "C15"),  # ce
        ]
        for i, (color, gtp) in enumerate(expected):
            m = real_game.moves[i]
            assert m.color == color, f"第{i+1}手颜色不匹配: {m.color} != {color}"
            assert m.gtp_coord == gtp, f"第{i+1}手坐标不匹配: {m.gtp_coord} != {gtp}"

    def test_alternating_colors(self, real_game):
        """绝大多数着手应黑白交替（末尾可能有计目阶段的例外）。"""
        alternation_breaks = 0
        for i in range(1, len(real_game.moves)):
            if real_game.moves[i].color == real_game.moves[i - 1].color:
                alternation_breaks += 1
        # Fox Weiqi SGF 末尾可能有少量连续同色手（计目），允许少量
        assert alternation_breaks <= 5, f"颜色交替中断次数过多: {alternation_breaks}"

    def test_no_invalid_coordinates(self, real_game):
        """所有非虚手的坐标应在 19 路棋盘范围内。"""
        for m in real_game.moves:
            if m.is_pass:
                continue
            assert m.row is not None and 0 <= m.row < 19, f"行越界: move {m.move_number}"
            assert m.col is not None and 0 <= m.col < 19, f"列越界: move {m.move_number}"

    def test_katago_format(self, real_game):
        """katago_moves() 应返回合法的 [color, coord] 列表。"""
        km = real_game.katago_moves()
        assert len(km) == real_game.total_moves
        for pair in km:
            assert len(pair) == 2
            assert pair[0] in ("B", "W")
            assert isinstance(pair[1], str) and len(pair[1]) >= 2

    # ── Reviewer 全盘分析 ────────────────────────────────

    @pytest.mark.asyncio
    async def test_full_review_structure(self, real_game, dynamic_engine, review_cfg):
        """完整复盘流程 — 验证结果结构完整且数值合理。"""
        reviewer = Reviewer(dynamic_engine, review_cfg, llm=None)

        progress_log = []

        async def on_progress(cur, total):
            progress_log.append((cur, total))

        result = await reviewer.review_game(
            real_game, with_comments=False, progress_callback=on_progress,
        )

        total = real_game.total_moves

        # 基本结构
        assert result.total_moves == total
        assert len(result.move_analyses) == total
        assert len(result.black_winrate_curve) == total + 1

        # 进度回调完整
        assert len(progress_log) == total + 1
        assert progress_log[-1] == (total + 1, total + 1)

        # 至少检测出一些问题手
        assert result.total_problems >= 1

        # 每手的基本字段
        for ma in result.move_analyses:
            assert 1 <= ma.move_number <= total
            assert ma.color in ("B", "W")
            assert 0.0 <= ma.winrate_before <= 1.0
            assert 0.0 <= ma.winrate_after <= 1.0
            assert 0.0 <= ma.black_winrate <= 1.0
            assert ma.territory_before is not None
            assert ma.territory_after is not None
            assert len(ma.best_variations) == review_cfg.top_variations

    @pytest.mark.asyncio
    async def test_problem_moves_severity(self, real_game, dynamic_engine, review_cfg):
        """问题手应有合法的严重程度标签。"""
        reviewer = Reviewer(dynamic_engine, review_cfg, llm=None)
        result = await reviewer.review_game(real_game, with_comments=False)

        valid_severities = {Severity.MINOR, Severity.QUESTIONABLE, Severity.BAD}
        for pm in result.problem_moves:
            assert pm.is_problem
            assert pm.severity in valid_severities
            assert pm.winrate_drop >= 0.02  # 至少 2% 跌幅

    @pytest.mark.asyncio
    async def test_result_json_serialization(self, real_game, dynamic_engine, review_cfg):
        """完整结果应能无损序列化为 JSON。"""
        reviewer = Reviewer(dynamic_engine, review_cfg, llm=None)
        result = await reviewer.review_game(real_game, with_comments=False)

        d = result.to_dict()
        json_str = json.dumps(d, ensure_ascii=False)
        reparsed = json.loads(json_str)

        assert reparsed["totalMoves"] == real_game.total_moves
        assert len(reparsed["moveAnalyses"]) == real_game.total_moves
        assert len(reparsed["blackWinrateCurve"]) == real_game.total_moves + 1

    @pytest.mark.asyncio
    async def test_review_with_llm(self, real_game, dynamic_engine, review_cfg):
        """含 LLM 解说时，只有阈值以上的问题手获得解说。"""
        llm = MockLLMService()
        reviewer = Reviewer(dynamic_engine, review_cfg, llm=llm)
        result = await reviewer.review_game(real_game, with_comments=True)

        commented = [m for m in result.problem_moves if m.comment]
        for ma in commented:
            assert ma.winrate_drop >= review_cfg.winrate_threshold
            assert len(ma.comment) > 10

    # ── API 端点 ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_api_upload_real_sgf(self, real_sgf_text):
        """通过 API 上传真实棋谱并验证解析结果。"""
        engine = DynamicMockKataGoEngine()
        llm = MockLLMService()
        with patch("main.engine", engine), patch("main.llm_service", llm):
            from main import app

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.post("/api/upload/text", json={"sgf": real_sgf_text})
                assert resp.status_code == 200
                data = resp.json()
                assert data["totalMoves"] > 200
                assert data["gameInfo"]["blackPlayer"] == "知更不咕鸟"
                assert data["gameInfo"]["whitePlayer"] == "朕肖"
                assert data["gameInfo"]["result"] == "W+8.25"

    @pytest.mark.asyncio
    async def test_api_full_review_real_game(self, real_sgf_text):
        """完整 API 流程: 上传真实棋谱 → 分析 → 获取结果。"""
        engine = DynamicMockKataGoEngine()
        llm = MockLLMService()
        with patch("main.engine", engine), patch("main.llm_service", llm):
            from main import app

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as c:
                # 上传
                upload = (await c.post(
                    "/api/upload/text", json={"sgf": real_sgf_text}
                )).json()
                gid = upload["gameId"]
                total = upload["totalMoves"]

                # 启动复盘
                resp = await c.post(f"/api/review/{gid}", json={"withComments": False})
                assert resp.status_code == 200

                # 轮询（200+ 手游戏需要更多时间）
                for _ in range(500):
                    await asyncio.sleep(0.05)
                    s = (await c.get(f"/api/review/{gid}/status")).json()
                    if s["status"] == "done":
                        break
                    if s["status"] == "error":
                        pytest.fail(f"分析出错: {s.get('error')}")
                else:
                    pytest.fail("分析超时未完成")

                # 获取结果
                result = (await c.get(f"/api/review/{gid}/result")).json()
                assert result["totalMoves"] == total
                assert len(result["moveAnalyses"]) == total
                assert len(result["blackWinrateCurve"]) == total + 1
                assert result["totalProblems"] >= 1

                # 获取问题手
                problems = (await c.get(f"/api/review/{gid}/problems")).json()
                assert problems["totalProblems"] >= 1

                # 获取胜率曲线
                wr = (await c.get(f"/api/review/{gid}/winrate")).json()
                assert len(wr["curve"]) == total + 1

                # 查询第一手
                m1 = (await c.get(f"/api/review/{gid}/move/1")).json()
                assert m1["moveNumber"] == 1
                assert m1["color"] == "B"
