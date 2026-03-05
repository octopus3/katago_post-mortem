"""KataGo Analysis Engine 封装 — 管理进程生命周期与 JSON 行通信

用法:
    engine = KataGoEngine(cfg.katago)
    await engine.start()
    result = await engine.query(moves=[["B","D4"],["W","Q16"]], komi=7.5)
    await engine.stop()
"""

import asyncio
import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import KataGoConfig

logger = logging.getLogger(__name__)


class KataGoEngine:
    """与 KataGo analysis 模式的异步通信封装。

    使用 subprocess.Popen + 后台线程读取 stdout，
    避免 Windows 上 asyncio 事件循环不支持子进程的问题。
    """

    def __init__(self, cfg: KataGoConfig):
        self._cfg = cfg
        self._process: Optional[subprocess.Popen] = None
        self._counter = 0
        self._pending: Dict[str, asyncio.Future] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── 生命周期 ──────────────────────────────────────────────

    async def start(self) -> None:
        exe = Path(self._cfg.executable)
        model = Path(self._cfg.model)
        if not exe.exists():
            raise FileNotFoundError(f"KataGo 可执行文件不存在: {exe}")
        if not model.exists():
            raise FileNotFoundError(f"KataGo 模型文件不存在: {model}")
        if self._cfg.config:
            cfg_path = Path(self._cfg.config)
            if not cfg_path.exists():
                raise FileNotFoundError(f"KataGo 配置文件不存在: {cfg_path}")

        cmd = [
            self._cfg.executable,
            "analysis",
            "-model", self._cfg.model,
        ]
        if self._cfg.config:
            cmd += ["-config", self._cfg.config]

        logger.info("启动 KataGo: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._loop = asyncio.get_running_loop()

        self._stderr_thread = threading.Thread(
            target=self._stderr_loop_sync, daemon=True, name="katago-stderr"
        )
        self._stderr_thread.start()

        await asyncio.sleep(2)

        ret = self._process.poll()
        if ret is not None:
            self._process = None
            raise RuntimeError(f"KataGo 进程启动后退出 (code={ret})，请查看上方 stderr 日志")

        self._reader_thread = threading.Thread(
            target=self._read_loop_sync, daemon=True, name="katago-reader"
        )
        self._reader_thread.start()
        logger.info("KataGo 进程已启动 (pid=%s)", self._process.pid)

    async def stop(self) -> None:
        if self._process is None:
            return
        logger.info("正在关闭 KataGo …")
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        self._process = None
        self._loop = None
        logger.info("KataGo 已关闭")

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ── 查询接口 ──────────────────────────────────────────────

    async def query(
        self,
        moves: List[List[str]],
        rules: str = "chinese",
        komi: float = 7.5,
        max_visits: Optional[int] = None,
        board_x_size: int = 19,
        board_y_size: int = 19,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """发送单次分析请求并等待结果。

        Args:
            moves: 着手序列，如 [["B","D4"], ["W","Q16"]]
            rules: 规则，chinese / japanese / korean 等
            komi: 贴目
            max_visits: 搜索次数，None 则用配置默认值
            board_x_size / board_y_size: 棋盘大小
            extra: 任意额外字段，会合并进请求 JSON

        Returns:
            KataGo 返回的完整 JSON 响应字典
        """
        if not self.is_running:
            raise RuntimeError("KataGo 未启动，请先调用 start()")

        self._counter += 1
        query_id = f"q_{self._counter}"

        komi = max(-150.0, min(150.0, round(komi * 2) / 2))

        request: Dict[str, Any] = {
            "id": query_id,
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_x_size,
            "boardYSize": board_y_size,
            "maxVisits": max_visits or self._cfg.max_visits,
            "overrideSettings": {
                "reportAnalysisWinratesAs": "SIDETOMOVE",
            },
        }
        if extra:
            request.update(extra)

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[query_id] = fut

        line = json.dumps(request, ensure_ascii=False) + "\n"
        logger.info(">>> %s", line.rstrip()[:300])
        await loop.run_in_executor(None, self._write_stdin, line)

        return await fut

    def _write_stdin(self, line: str) -> None:
        self._process.stdin.write(line.encode("utf-8"))
        self._process.stdin.flush()

    # ── 后台读取线程 ──────────────────────────────────────────

    def _stderr_loop_sync(self) -> None:
        """持续读取 stderr 并输出到日志，方便排查 KataGo 启动/运行问题。"""
        try:
            while self._process and self._process.poll() is None:
                raw = self._process.stderr.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    logger.info("[KataGo stderr] %s", line)
        except Exception:
            pass

    def _read_loop_sync(self) -> None:
        """在独立线程中从 stdout 读取 JSON 行，通过 call_soon_threadsafe 分派结果。"""
        try:
            while True:
                raw = self._process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                logger.info("<<< %s", line[:500])
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("无法解析 KataGo 输出: %s", line[:200])
                    continue

                qid = data.get("id")
                if qid and qid in self._pending:
                    fut = self._pending.pop(qid)
                    if "error" in data:
                        self._loop.call_soon_threadsafe(
                            fut.set_exception,
                            RuntimeError(f"KataGo 错误: {data['error']}"),
                        )
                    elif not fut.done():
                        self._loop.call_soon_threadsafe(fut.set_result, data)
        except Exception:
            logger.exception("KataGo 读取线程异常")


# ── 独立测试入口 ──────────────────────────────────────────────

async def _test():
    """简单测试: 发送一个空棋盘查询，打印 KataGo 推荐的前 3 手。"""
    from config import load_config
    cfg = load_config()

    engine = KataGoEngine(cfg.katago)
    await engine.start()

    try:
        print("发送查询: 空棋盘, 中国规则, 贴目 7.5 …")
        result = await engine.query(moves=[], komi=7.5)

        root = result.get("rootInfo", {})
        print(f"\n当前胜率: 黑 {root.get('winrate', 0):.1%}")
        print(f"当前分数: 黑 {root.get('scoreLead', 0):+.1f}")

        print("\nKataGo 推荐着手:")
        for i, info in enumerate(result.get("moveInfos", [])[:3]):
            move = info.get("move", "?")
            visits = info.get("visits", 0)
            winrate = info.get("winrate", 0)
            score = info.get("scoreLead", 0)
            pv = " ".join(info.get("pv", [])[:5])
            print(f"  {i+1}. {move}  胜率={winrate:.1%}  分数={score:+.1f}  "
                  f"访问={visits}  变化={pv}")
    finally:
        await engine.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    asyncio.run(_test())
