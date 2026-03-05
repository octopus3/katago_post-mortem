"""FastAPI 入口 — KataGo 复盘 Web 服务

提供 SGF 上传、KataGo 分析、结果查询、LLM 解说等完整 API。
支持 WebSocket 实时推送分析进度。
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import load_config
from katago_engine import KataGoEngine
from llm_service import LLMService
from reviewer import MoveAnalysis, ReviewResult, Reviewer
from sgf_parser import ParsedGame, parse_sgf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── 全局配置 & 服务实例 ────────────────────────────────────────

cfg = load_config()
engine = KataGoEngine(cfg.katago)
llm_service = LLMService(cfg.llm)

# ── 内存存储 ──────────────────────────────────────────────────


class GameStatus(str, Enum):
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    DONE = "done"
    ERROR = "error"


@dataclass
class GameSession:
    game_id: str
    sgf_text: str
    parsed: ParsedGame
    status: GameStatus = GameStatus.UPLOADED
    progress_current: int = 0
    progress_total: int = 0
    result: Optional[ReviewResult] = None
    error: Optional[str] = None
    ws_clients: List[WebSocket] = field(default_factory=list)
    cancel_event: Optional[asyncio.Event] = None


_games: Dict[str, GameSession] = {}

# ── Pydantic 响应模型 ─────────────────────────────────────────


class UploadResponse(BaseModel):
    gameId: str
    gameInfo: Dict[str, Any]
    totalMoves: int


class StatusResponse(BaseModel):
    gameId: str
    status: str
    progressCurrent: int
    progressTotal: int
    error: Optional[str] = None


class ReviewStartResponse(BaseModel):
    gameId: str
    message: str


class MoveAnalysisResponse(BaseModel):
    moveNumber: int
    color: str
    gtpCoord: str
    winrateBefore: float
    winrateAfter: float
    winrateDrop: float
    scoreBefore: float
    scoreAfter: float
    scoreDrop: float
    blackWinrate: float
    bestVariations: List[Dict[str, Any]]
    isProblem: bool
    severity: Optional[str]
    comment: str


class ReviewRequest(BaseModel):
    withComments: bool = True


class CommentRequest(BaseModel):
    force: bool = False


# ── 应用生命周期 ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动 KataGo 引擎 …")
    try:
        await engine.start()
        logger.info("KataGo 引擎已就绪")
    except Exception as e:
        logger.warning("KataGo 启动失败 [%s: %s]，分析功能将不可用。请检查 config.yaml 中的路径配置。",
                       type(e).__name__, e)
    yield
    logger.info("关闭 KataGo 引擎 …")
    await engine.stop()


app = FastAPI(title="KataGo Review", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ── 工具函数 ──────────────────────────────────────────────────


def _get_session(game_id: str) -> GameSession:
    session = _games.get(game_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"棋谱不存在: {game_id}")
    return session


def _game_info_dict(parsed: ParsedGame) -> Dict[str, Any]:
    info = parsed.info
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


# ── API 路由 ──────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "katagoRunning": engine.is_running,
        "gamesLoaded": len(_games),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_sgf(file: UploadFile = File(...)):
    """上传并解析 SGF 棋谱文件。"""
    if not file.filename or not file.filename.lower().endswith(".sgf"):
        raise HTTPException(status_code=400, detail="请上传 .sgf 格式的棋谱文件")

    content = await file.read()
    try:
        sgf_text = content.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="文件编码无法识别")

    try:
        parsed = parse_sgf(sgf_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"SGF 解析失败: {e}")

    game_id = uuid.uuid4().hex[:12]
    _games[game_id] = GameSession(
        game_id=game_id,
        sgf_text=sgf_text,
        parsed=parsed,
    )

    logger.info(
        "棋谱已上传: %s — %s vs %s (%d 手)",
        game_id, parsed.info.black_player, parsed.info.white_player, parsed.total_moves,
    )

    return UploadResponse(
        gameId=game_id,
        gameInfo=_game_info_dict(parsed),
        totalMoves=parsed.total_moves,
    )


@app.post("/api/upload/text", response_model=UploadResponse)
async def upload_sgf_text(body: dict):
    """通过 JSON 正文直接提交 SGF 文本（方便调试）。"""
    sgf_text = body.get("sgf", "").strip()
    if not sgf_text:
        raise HTTPException(status_code=400, detail="缺少 sgf 字段")

    try:
        parsed = parse_sgf(sgf_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"SGF 解析失败: {e}")

    game_id = uuid.uuid4().hex[:12]
    _games[game_id] = GameSession(
        game_id=game_id,
        sgf_text=sgf_text,
        parsed=parsed,
    )

    return UploadResponse(
        gameId=game_id,
        gameInfo=_game_info_dict(parsed),
        totalMoves=parsed.total_moves,
    )


@app.get("/api/games")
async def list_games():
    """列出所有已加载的棋谱及其状态。"""
    return [
        {
            "gameId": s.game_id,
            "blackPlayer": s.parsed.info.black_player,
            "whitePlayer": s.parsed.info.white_player,
            "totalMoves": s.parsed.total_moves,
            "status": s.status.value,
        }
        for s in _games.values()
    ]


@app.get("/api/game/{game_id}")
async def get_game_info(game_id: str):
    """获取棋谱的基本信息和棋步列表。"""
    session = _get_session(game_id)
    moves = [
        {
            "moveNumber": m.move_number,
            "color": m.color,
            "gtpCoord": m.gtp_coord,
            "isPass": m.is_pass,
        }
        for m in session.parsed.moves
    ]
    return {
        "gameId": game_id,
        "gameInfo": _game_info_dict(session.parsed),
        "totalMoves": session.parsed.total_moves,
        "moves": moves,
        "status": session.status.value,
    }


@app.post("/api/review/{game_id}", response_model=ReviewStartResponse)
async def start_review(game_id: str, body: ReviewRequest = ReviewRequest()):
    """启动 KataGo 复盘分析（异步后台运行）。"""
    session = _get_session(game_id)

    if not engine.is_running:
        raise HTTPException(status_code=503, detail="KataGo 引擎未运行，请检查配置")

    if session.status == GameStatus.ANALYZING:
        raise HTTPException(status_code=409, detail="该棋谱正在分析中，请等待完成")

    session.status = GameStatus.ANALYZING
    session.progress_current = 0
    session.progress_total = session.parsed.total_moves + 1
    session.result = None
    session.error = None
    session.cancel_event = asyncio.Event()

    asyncio.create_task(_run_review(session, with_comments=body.withComments))

    return ReviewStartResponse(
        gameId=game_id,
        message=f"复盘分析已启动，共需分析 {session.progress_total} 个局面",
    )


@app.post("/api/review/{game_id}/stop")
async def stop_review(game_id: str):
    """中止正在进行的复盘分析，已完成的部分将生成结果。"""
    session = _get_session(game_id)
    if session.status != GameStatus.ANALYZING:
        raise HTTPException(status_code=400, detail="当前没有进行中的分析")
    if session.cancel_event is None:
        raise HTTPException(status_code=400, detail="无法取消该分析")

    session.cancel_event.set()
    logger.info("收到停止请求: %s (已完成 %d/%d)", game_id, session.progress_current, session.progress_total)
    return {"gameId": game_id, "message": "已发送停止信号，正在生成部分结果…"}


async def _run_review(session: GameSession, with_comments: bool = True):
    """后台执行复盘分析，通过 WebSocket 推送进度。"""
    reviewer = Reviewer(engine, cfg.review, llm_service if with_comments else None)

    async def on_progress(current: int, total: int):
        session.progress_current = current
        session.progress_total = total
        await _broadcast_ws(session, {
            "type": "progress",
            "current": current,
            "total": total,
        })

    try:
        result = await reviewer.review_game(
            session.parsed,
            with_comments=with_comments,
            progress_callback=on_progress,
            cancel_event=session.cancel_event,
        )
        session.result = result
        cancelled = session.cancel_event is not None and session.cancel_event.is_set()
        session.status = GameStatus.DONE
        await _broadcast_ws(session, {
            "type": "done",
            "totalProblems": result.total_problems,
            "partial": cancelled,
            "analyzedMoves": result.total_moves,
        })
        if cancelled:
            logger.info("复盘被中止: %s (已分析 %d/%d 手, %d 个问题手)",
                        session.game_id, result.total_moves, session.parsed.total_moves, result.total_problems)
        else:
            logger.info("复盘完成: %s (%d 个问题手)", session.game_id, result.total_problems)
    except Exception as e:
        session.status = GameStatus.ERROR
        session.error = str(e)
        await _broadcast_ws(session, {"type": "error", "message": str(e)})
        logger.exception("复盘失败: %s", session.game_id)


async def _broadcast_ws(session: GameSession, data: dict):
    """向该棋谱的所有 WebSocket 客户端广播消息。"""
    disconnected = []
    for ws in session.ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        session.ws_clients.remove(ws)


@app.get("/api/review/{game_id}/status", response_model=StatusResponse)
async def get_review_status(game_id: str):
    """查询复盘进度。"""
    session = _get_session(game_id)
    return StatusResponse(
        gameId=game_id,
        status=session.status.value,
        progressCurrent=session.progress_current,
        progressTotal=session.progress_total,
        error=session.error,
    )


@app.get("/api/review/{game_id}/result")
async def get_review_result(game_id: str):
    """获取完整复盘结果。"""
    session = _get_session(game_id)
    if session.status == GameStatus.UPLOADED:
        raise HTTPException(status_code=400, detail="尚未启动分析")
    if session.status == GameStatus.ANALYZING:
        raise HTTPException(status_code=202, detail="分析进行中，请稍后再试")
    if session.status == GameStatus.ERROR:
        raise HTTPException(status_code=500, detail=f"分析出错: {session.error}")
    return session.result.to_dict()


@app.get("/api/review/{game_id}/move/{move_number}")
async def get_move_analysis(game_id: str, move_number: int):
    """获取单手的分析详情。"""
    session = _get_session(game_id)
    if not session.result:
        raise HTTPException(status_code=400, detail="尚无分析结果")

    if move_number < 1 or move_number > session.result.total_moves:
        raise HTTPException(
            status_code=404,
            detail=f"手数超出范围 (1-{session.result.total_moves})",
        )

    ma = session.result.move_analyses[move_number - 1]
    return ma.to_dict()


@app.get("/api/review/{game_id}/problems")
async def get_problem_moves(game_id: str):
    """获取所有问题手列表。"""
    session = _get_session(game_id)
    if not session.result:
        raise HTTPException(status_code=400, detail="尚无分析结果")

    return {
        "gameId": game_id,
        "totalProblems": session.result.total_problems,
        "problems": [m.to_dict() for m in session.result.problem_moves],
    }


@app.get("/api/review/{game_id}/winrate")
async def get_winrate_curve(game_id: str):
    """获取胜率曲线数据。"""
    session = _get_session(game_id)
    if not session.result:
        raise HTTPException(status_code=400, detail="尚无分析结果")

    return {
        "gameId": game_id,
        "initialBlackWinrate": round(session.result.initial_black_winrate, 4),
        "curve": [round(w, 4) for w in session.result.black_winrate_curve],
    }


@app.post("/api/review/{game_id}/comment/{move_number}")
async def generate_comment(game_id: str, move_number: int, body: CommentRequest = CommentRequest()):
    """为指定手数生成 / 重新生成 LLM 解说。"""
    session = _get_session(game_id)
    if not session.result:
        raise HTTPException(status_code=400, detail="尚无分析结果")

    if move_number < 1 or move_number > session.result.total_moves:
        raise HTTPException(
            status_code=404,
            detail=f"手数超出范围 (1-{session.result.total_moves})",
        )

    ma = session.result.move_analyses[move_number - 1]

    if ma.comment and not body.force:
        return {"moveNumber": move_number, "comment": ma.comment, "cached": True}

    reviewer = Reviewer(engine, cfg.review, llm_service)
    try:
        prompt = reviewer._build_comment_prompt(session.parsed, ma)
        comment = (await llm_service.chat(prompt)).strip()
        ma.comment = comment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 解说生成失败: {e}")

    return {"moveNumber": move_number, "comment": ma.comment, "cached": False}


# ── WebSocket 端点 ────────────────────────────────────────────


@app.websocket("/ws/review/{game_id}")
async def ws_review_progress(websocket: WebSocket, game_id: str):
    """WebSocket 端点，实时推送分析进度和完成通知。

    客户端连接后立即收到当前状态快照，之后在分析过程中持续收到进度更新。
    """
    session = _games.get(game_id)
    if not session:
        await websocket.close(code=4004, reason="棋谱不存在")
        return

    await websocket.accept()
    session.ws_clients.append(websocket)
    logger.info("WebSocket 已连接: %s (客户端数=%d)", game_id, len(session.ws_clients))

    await websocket.send_json({
        "type": "status",
        "status": session.status.value,
        "progressCurrent": session.progress_current,
        "progressTotal": session.progress_total,
    })

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in session.ws_clients:
            session.ws_clients.remove(websocket)
        logger.info("WebSocket 已断开: %s", game_id)


# ── 前端页面路由 ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端主页面。"""
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>KataGo Review</h1><p>前端文件未找到</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


# ── 启动入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
    )
