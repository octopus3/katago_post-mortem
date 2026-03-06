/**
 * Main application controller — coordinates board, chart, and info panels.
 */

(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────

    const S = {
        gameId: null,
        gameInfo: null,
        moves: [],
        totalMoves: 0,
        currentMove: 0,
        status: 'idle',      // idle | loaded | analyzing | reviewed
        result: null,
        moveAnalyses: [],
        problemMoves: [],
        winrateCurve: [],
        activeVariation: -1,
        ws: null,
    };

    // ── DOM refs ───────────────────────────────────────────

    const $ = (sel) => document.querySelector(sel);
    const uploadOverlay = $('#upload-overlay');
    const dropZone = $('#drop-zone');
    const fileInput = $('#file-input');
    const uploadError = $('#upload-error');

    const appEl = $('#app');
    const progressSection = $('#progress-section');
    const progressBar = $('#progress-bar');
    const progressText = $('#progress-text');

    const gameInfoHeader = $('#game-info-header');
    const gameInfoBody = $('#game-info-body');
    const moveIndicator = $('#move-indicator');
    const moveSlider = $('#move-slider');
    const reviewActions = $('#review-actions');

    const analysisBody = $('#analysis-body');
    const severityBadge = $('#severity-badge');
    const variationsBody = $('#variations-body');
    const commentBody = $('#comment-body');
    const problemsBody = $('#problems-body');
    const problemCount = $('#problem-count');

    const boardCanvas = $('#board-canvas');
    const chartCanvas = $('#winrate-chart');

    // ── Components ─────────────────────────────────────────

    const board = new GoBoard(boardCanvas);
    const chart = new WinrateChart(chartCanvas);

    chart.onMoveClick = (idx) => goToMove(idx);

    // ── Throttle helper ───────────────────────────────────

    function throttle(fn, ms) {
        let last = 0, timer = null;
        return function (...args) {
            const now = Date.now();
            const remaining = ms - (now - last);
            clearTimeout(timer);
            if (remaining <= 0) {
                last = now;
                fn.apply(this, args);
            } else {
                timer = setTimeout(() => {
                    last = Date.now();
                    fn.apply(this, args);
                }, remaining);
            }
        };
    }

    // ── API helper ─────────────────────────────────────────

    async function api(url, opts = {}) {
        const resp = await fetch(url, opts);
        if (!resp.ok) {
            const body = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(body.detail || 'API 请求失败');
        }
        return resp.json();
    }

    // ── Upload ─────────────────────────────────────────────

    function showUploadError(msg) {
        uploadError.textContent = msg;
        uploadError.classList.remove('hidden');
    }

    async function handleFile(file) {
        if (!file || !file.name.toLowerCase().endsWith('.sgf')) {
            showUploadError('请选择 .sgf 格式的棋谱文件');
            return;
        }
        uploadError.classList.add('hidden');

        const form = new FormData();
        form.append('file', file);
        try {
            const data = await api('/api/upload', { method: 'POST', body: form });
            S.gameId = data.gameId;
            S.gameInfo = data.gameInfo;
            S.totalMoves = data.totalMoves;
            await loadGame();
        } catch (e) {
            showUploadError(e.message);
        }
    }

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    // ── Load Game ──────────────────────────────────────────

    async function loadGame() {
        const data = await api(`/api/game/${S.gameId}`);
        S.moves = data.moves;
        S.totalMoves = data.totalMoves;
        S.gameInfo = data.gameInfo;
        S.status = data.status === 'done' ? 'reviewed' : 'loaded';
        S.currentMove = 0;

        uploadOverlay.classList.add('fade-out');
        appEl.classList.remove('hidden');

        renderGameInfo();
        moveSlider.max = S.totalMoves;
        moveSlider.value = 0;

        board.size = S.gameInfo.boardSize || 19;
        board.resize();
        goToMove(0);

        if (S.status === 'reviewed') {
            await loadResults();
        } else {
            reviewActions.classList.remove('hidden');
        }
    }

    function renderGameInfo() {
        const g = S.gameInfo;
        gameInfoHeader.textContent =
            `${g.blackPlayer || '?'} (黑) vs ${g.whitePlayer || '?'} (白)` +
            (g.result ? `  ${g.result}` : '');

        const items = [
            ['黑方', `${g.blackPlayer || '?'} ${g.blackRank || ''}`],
            ['白方', `${g.whitePlayer || '?'} ${g.whiteRank || ''}`],
            ['结果', g.result || '-'],
            ['贴目', g.komi],
            ['路数', `${g.boardSize}×${g.boardSize}`],
            ['规则', g.rules || '-'],
        ];
        if (g.date) items.push(['日期', g.date]);
        if (g.event) items.push(['赛事', g.event]);

        gameInfoBody.innerHTML = '<div class="game-info-grid">' +
            items.map(([l, v]) =>
                `<div class="info-item"><span class="info-label">${l}</span><span class="info-value">${v}</span></div>`
            ).join('') + '</div>';
    }

    // ── Board position rebuild ─────────────────────────────

    function buildPosition(upTo) {
        const logic = new GoLogic(S.gameInfo.boardSize || 19);
        const hist = [];
        for (let i = 0; i < upTo && i < S.moves.length; i++) {
            const m = S.moves[i];
            if (m.isPass) { hist.push(m); continue; }
            const pos = gtpToXY(m.gtpCoord, logic.size);
            if (pos) {
                logic.playMove(pos.x, pos.y, m.color === 'B' ? 1 : 2);
            }
            hist.push(m);
        }
        return { board: logic.board.map(r => [...r]), history: hist };
    }

    // ── Navigation ─────────────────────────────────────────

    const goToMove = throttle(function (n) {
        n = Math.max(0, Math.min(n, S.totalMoves));
        S.currentMove = n;
        S.activeVariation = -1;

        const { board: pos, history } = buildPosition(n);
        let last = null;
        if (n > 0) {
            const m = S.moves[n - 1];
            if (!m.isPass) {
                const p = gtpToXY(m.gtpCoord, S.gameInfo.boardSize || 19);
                if (p) last = p;
            }
        }

        let bestGtp = null;
        if (S.showBestMove && S.moveAnalyses[n - 1]) {
            const vars = S.moveAnalyses[n - 1].bestVariations;
            if (vars && vars.length) bestGtp = vars[0].move;
        }

        board.moveHistory = history;
        board.updateState(pos, last, bestGtp);

        moveIndicator.textContent = `${n} / ${S.totalMoves}`;
        moveSlider.value = n;
        chart.highlightMove(n);
        updateAnalysisPanel();
    }, 50);

    S.showBestMove = false;

    $('#btn-first').addEventListener('click', () => goToMove(0));
    $('#btn-prev10').addEventListener('click', () => goToMove(S.currentMove - 10));
    $('#btn-prev').addEventListener('click', () => goToMove(S.currentMove - 1));
    $('#btn-next').addEventListener('click', () => goToMove(S.currentMove + 1));
    $('#btn-next10').addEventListener('click', () => goToMove(S.currentMove + 10));
    $('#btn-last').addEventListener('click', () => goToMove(S.totalMoves));

    moveSlider.addEventListener('input', () => goToMove(parseInt(moveSlider.value, 10)));

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
        switch (e.key) {
            case 'ArrowLeft':  e.preventDefault(); goToMove(S.currentMove - (e.shiftKey ? 10 : 1)); break;
            case 'ArrowRight': e.preventDefault(); goToMove(S.currentMove + (e.shiftKey ? 10 : 1)); break;
            case 'Home':       e.preventDefault(); goToMove(0); break;
            case 'End':        e.preventDefault(); goToMove(S.totalMoves); break;
            case 'Escape':     board.clearVariation(); S.activeVariation = -1; renderVariations(); break;
        }
    });

    // ── Board options ──────────────────────────────────────

    $('#chk-coords').addEventListener('change', (e) => {
        board.showCoords = e.target.checked;
        board.resize();
    });
    $('#chk-numbers').addEventListener('change', (e) => {
        board.showNumbers = e.target.checked;
        board.draw();
    });
    $('#chk-best-move').addEventListener('change', (e) => {
        S.showBestMove = e.target.checked;
        board.showBestMove = e.target.checked;
        goToMove(S.currentMove);
    });

    // ── Start / Stop Review ─────────────────────────────────

    const btnStopReview = $('#btn-stop-review');

    $('#btn-start-review').addEventListener('click', async () => {
        if (!S.gameId) return;
        const withComments = $('#chk-with-comments').checked;
        try {
            await api(`/api/review/${S.gameId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ withComments }),
            });
            S.status = 'analyzing';
            reviewActions.classList.add('hidden');
            progressSection.classList.remove('hidden');
            btnStopReview.disabled = false;
            btnStopReview.textContent = '停止分析';
            connectWS();
        } catch (e) {
            alert('启动分析失败: ' + e.message);
        }
    });

    btnStopReview.addEventListener('click', async () => {
        if (!S.gameId || S.status !== 'analyzing') return;
        btnStopReview.disabled = true;
        btnStopReview.textContent = '正在停止…';
        try {
            await api(`/api/review/${S.gameId}/stop`, { method: 'POST' });
            progressText.textContent = '正在停止，生成已分析部分的结果…';
        } catch (e) {
            btnStopReview.disabled = false;
            btnStopReview.textContent = '停止分析';
            alert('停止失败: ' + e.message);
        }
    });

    // ── WebSocket ──────────────────────────────────────────

    function connectWS() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${proto}//${location.host}/ws/review/${S.gameId}`;
        const ws = new WebSocket(url);
        S.ws = ws;

        ws.onmessage = async (ev) => {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'progress') {
                const pct = msg.total ? ((msg.current / msg.total) * 100) : 0;
                progressBar.style.width = pct + '%';
                progressText.textContent = `分析中 ${msg.current} / ${msg.total}`;
            } else if (msg.type === 'done') {
                btnStopReview.disabled = true;
                if (msg.partial) {
                    progressBar.style.width = '100%';
                    progressText.textContent = `分析已停止，已分析 ${msg.analyzedMoves} 手，发现 ${msg.totalProblems} 个问题手`;
                } else {
                    progressBar.style.width = '100%';
                    progressText.textContent = `分析完成！共发现 ${msg.totalProblems} 个问题手`;
                }
                setTimeout(() => progressSection.classList.add('hidden'), 2500);
                S.status = 'reviewed';
                await loadResults();
            } else if (msg.type === 'error') {
                btnStopReview.disabled = true;
                progressText.textContent = '分析出错: ' + msg.message;
                progressBar.style.width = '0';
            }
        };

        ws.onerror = () => {
            progressText.textContent = 'WebSocket 连接出错，尝试轮询…';
            startPolling();
        };

        ws.onclose = () => {
            if (S.status === 'analyzing') startPolling();
        };
    }

    function startPolling() {
        const poll = setInterval(async () => {
            try {
                const st = await api(`/api/review/${S.gameId}/status`);
                const pct = st.progressTotal ? ((st.progressCurrent / st.progressTotal) * 100) : 0;
                progressBar.style.width = pct + '%';
                progressText.textContent = `分析中 ${st.progressCurrent} / ${st.progressTotal}`;

                if (st.status === 'done') {
                    clearInterval(poll);
                    progressBar.style.width = '100%';
                    progressText.textContent = '分析完成！';
                    setTimeout(() => progressSection.classList.add('hidden'), 2000);
                    S.status = 'reviewed';
                    await loadResults();
                } else if (st.status === 'error') {
                    clearInterval(poll);
                    progressText.textContent = '分析出错: ' + st.error;
                }
            } catch (_) { /* retry */ }
        }, 2000);
    }

    // ── Load Results ───────────────────────────────────────

    async function loadResults() {
        try {
            const data = await api(`/api/review/${S.gameId}/result`);
            S.result = data;
            S.moveAnalyses = data.moveAnalyses || [];
            S.problemMoves = data.problemMoves || [];
            S.winrateCurve = data.blackWinrateCurve || [];
            S.whiteWinrateCurve = data.whiteWinrateCurve || [];

            chart.setData(S.winrateCurve, S.whiteWinrateCurve, S.problemMoves);
            renderProblemList();
            reviewActions.classList.add('hidden');
            btnSaveReview.classList.remove('hidden');
            goToMove(S.currentMove);
        } catch (e) {
            console.error('加载结果失败:', e);
        }
    }

    // ── Chart View Toggle ─────────────────────────────────

    document.querySelectorAll('#chart-view-btns .view-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#chart-view-btns .view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            chart.setViewMode(btn.dataset.view);
        });
    });

    // ── Analysis Panel ─────────────────────────────────────

    function updateAnalysisPanel() {
        const n = S.currentMove;
        if (!n || !S.moveAnalyses.length || n > S.moveAnalyses.length) {
            analysisBody.innerHTML = '<div class="placeholder">选择一手棋查看分析</div>';
            severityBadge.classList.add('hidden');
            variationsBody.innerHTML = '<div class="placeholder">完成分析后查看推荐变化</div>';
            commentBody.innerHTML = '<div class="placeholder">问题手将自动获得 AI 评价</div>';
            return;
        }

        const ma = S.moveAnalyses[n - 1];
        if (!ma) {
            analysisBody.innerHTML = '<div class="placeholder">该手无分析数据</div>';
            return;
        }

        const colorCN = ma.color === 'B' ? '黑' : '白';
        const wrPct = (v) => (v * 100).toFixed(1) + '%';
        const dropCls = ma.winrateDrop > 0.02 ? 'negative' : (ma.winrateDrop < -0.02 ? 'positive' : '');
        const scoreCls = ma.scoreDrop > 0.5 ? 'negative' : (ma.scoreDrop < -0.5 ? 'positive' : '');

        analysisBody.innerHTML = `
            <div style="margin-bottom:8px;font-weight:600">
                第 ${ma.moveNumber} 手 ${colorCN} ${ma.gtpCoord}
            </div>
            <div class="analysis-grid">
                <div class="analysis-stat">
                    <div class="stat-label">胜率</div>
                    <div class="stat-value">${wrPct(ma.winrateAfter)}</div>
                    <div class="stat-sub">${wrPct(ma.winrateBefore)} → ${wrPct(ma.winrateAfter)}</div>
                </div>
                <div class="analysis-stat">
                    <div class="stat-label">胜率变化</div>
                    <div class="stat-value ${dropCls}">${ma.winrateDrop > 0 ? '-' : '+'}${wrPct(Math.abs(ma.winrateDrop))}</div>
                </div>
                <div class="analysis-stat">
                    <div class="stat-label">目差</div>
                    <div class="stat-value">${ma.scoreAfter > 0 ? '+' : ''}${ma.scoreAfter.toFixed(1)}</div>
                    <div class="stat-sub">${ma.scoreBefore > 0 ? '+' : ''}${ma.scoreBefore.toFixed(1)} → ${ma.scoreAfter > 0 ? '+' : ''}${ma.scoreAfter.toFixed(1)}</div>
                </div>
                <div class="analysis-stat">
                    <div class="stat-label">目差变化</div>
                    <div class="stat-value ${scoreCls}">${ma.scoreDrop > 0 ? '-' : '+'}${Math.abs(ma.scoreDrop).toFixed(1)}</div>
                </div>
            </div>`;

        if (ma.isProblem && ma.severity) {
            const sevMap = { vulgar: ['俗手', 'badge-vulgar'], slow: ['缓手', 'badge-slow'], minor: ['小疑问手', 'badge-minor'], questionable: ['疑问手', 'badge-questionable'], bad: ['恶手', 'badge-bad'] };
            const [text, cls] = sevMap[ma.severity] || ['问题手', 'badge-bad'];
            severityBadge.textContent = text;
            severityBadge.className = `badge ${cls}`;
            severityBadge.classList.remove('hidden');
        } else {
            severityBadge.classList.add('hidden');
        }

        renderVariations(ma);
        renderComment(ma);
    }

    // ── Variations ─────────────────────────────────────────

    function renderVariations(ma) {
        if (!ma || !ma.bestVariations || !ma.bestVariations.length) {
            variationsBody.innerHTML = '<div class="placeholder">无推荐变化</div>';
            return;
        }

        const items = ma.bestVariations.map((v, i) => {
            const rankCls = i < 3 ? `var-rank-${i + 1}` : '';
            const activeCls = S.activeVariation === i ? 'active' : '';
            const pvStr = v.pv ? v.pv.slice(0, 6).join(' → ') : '';
            return `<div class="var-item ${activeCls}" data-idx="${i}">
                <div class="var-rank ${rankCls}">${i + 1}</div>
                <div>
                    <div style="display:flex;gap:8px;align-items:baseline">
                        <span class="var-move">${v.move}</span>
                        <span class="var-stats">胜率 ${(v.winrate * 100).toFixed(1)}%　目差 ${v.scoreLead > 0 ? '+' : ''}${v.scoreLead.toFixed(1)}　访问 ${v.visits}</span>
                    </div>
                    ${pvStr ? `<div class="var-pv">${pvStr}</div>` : ''}
                </div>
            </div>`;
        });

        variationsBody.innerHTML = `<div class="var-list">${items.join('')}</div>`;

        variationsBody.querySelectorAll('.var-item').forEach(el => {
            el.addEventListener('click', () => {
                const idx = parseInt(el.dataset.idx, 10);
                toggleVariation(idx, ma);
            });
        });
    }

    function toggleVariation(idx, ma) {
        if (S.activeVariation === idx) {
            board.clearVariation();
            S.activeVariation = -1;
            renderVariations(ma);
            return;
        }
        S.activeVariation = idx;
        const v = ma.bestVariations[idx];
        if (!v || !v.pv) return;

        const startColor = ma.color;
        const varMoves = v.pv.map((gtp, i) => ({
            gtp,
            color: ((startColor === 'B') === (i % 2 === 0)) ? 'B' : 'W',
        }));
        board.setVariation(varMoves);
        renderVariations(ma);
    }

    // ── Comment ────────────────────────────────────────────

    function renderComment(ma) {
        if (!ma) {
            commentBody.innerHTML = '<div class="placeholder">问题手将自动获得 AI 评价</div>';
            return;
        }
        if (ma.comment) {
            const paragraphs = ma.comment.split(/\n\n|\n/).filter(p => p.trim());
            commentBody.innerHTML = paragraphs.map(p => `<p class="comment-text">${escapeHtml(p)}</p>`).join('');
        } else if (ma.isProblem) {
            commentBody.innerHTML = '<div class="placeholder">点击「生成评价」获取 AI 分析</div>';
        } else {
            commentBody.innerHTML = '<div class="placeholder">该手无需评价</div>';
        }
    }

    $('#btn-gen-comment').addEventListener('click', async () => {
        const n = S.currentMove;
        if (!n || !S.gameId || !S.moveAnalyses.length) return;
        commentBody.innerHTML = '<div class="comment-loading">正在生成 AI 评价…</div>';
        try {
            const data = await api(`/api/review/${S.gameId}/comment/${n}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ force: true }),
            });
            if (S.moveAnalyses[n - 1]) {
                S.moveAnalyses[n - 1].comment = data.comment;
            }
            renderComment(S.moveAnalyses[n - 1]);
        } catch (e) {
            commentBody.innerHTML = `<div class="placeholder" style="color:var(--danger)">生成失败: ${escapeHtml(e.message)}</div>`;
        }
    });

    // ── Problem List ───────────────────────────────────────

    function renderProblemList() {
        if (!S.problemMoves.length) {
            problemsBody.innerHTML = '<div class="placeholder">未发现问题手</div>';
            problemCount.classList.add('hidden');
            return;
        }

        problemCount.textContent = S.problemMoves.length;
        problemCount.classList.remove('hidden');

        const sevCN = { vulgar: '俗手', slow: '缓手', minor: '小疑问', questionable: '疑问手', bad: '恶手' };

        const items = S.problemMoves.map(pm => {
            const ma = S.moveAnalyses[pm.moveNumber - 1];
            if (!ma) return '';
            const colorCN = ma.color === 'B' ? '黑' : '白';
            const sev = sevCN[pm.severity] || '问题';
            const sevCls = `sev-${pm.severity || 'bad'}`;
            return `<div class="problem-item" data-move="${pm.moveNumber}">
                <span class="problem-move-num">${colorCN} 第${pm.moveNumber}手</span>
                <span class="problem-coord">${ma.gtpCoord}</span>
                <span class="problem-severity ${sevCls}">${sev}</span>
                <span class="problem-drop">-${(ma.winrateDrop * 100).toFixed(1)}%</span>
            </div>`;
        });

        problemsBody.innerHTML = items.join('');

        problemsBody.querySelectorAll('.problem-item').forEach(el => {
            el.addEventListener('click', () => {
                const move = parseInt(el.dataset.move, 10);
                goToMove(move);
            });
        });
    }

    // ── Save / Import Review ────────────────────────────────

    const btnSaveReview = $('#btn-save-review');
    const importInput = $('#import-input');

    function exportResults() {
        if (!S.result || !S.moveAnalyses.length) {
            alert('没有可保存的复盘结果');
            return;
        }

        const payload = {
            version: 1,
            exportedAt: new Date().toISOString(),
            gameInfo: S.gameInfo,
            totalMoves: S.totalMoves,
            moves: S.moves,
            moveAnalyses: S.moveAnalyses,
            problemMoves: S.problemMoves,
            blackWinrateCurve: S.winrateCurve,
            whiteWinrateCurve: S.whiteWinrateCurve,
        };

        const json = JSON.stringify(payload, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const black = S.gameInfo.blackPlayer || 'B';
        const white = S.gameInfo.whitePlayer || 'W';
        const date = S.gameInfo.date || new Date().toISOString().slice(0, 10);
        const filename = `复盘_${black}_vs_${white}_${date}.json`;

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function importResults(file) {
        if (!file || !file.name.toLowerCase().endsWith('.json')) {
            showUploadError('请选择 .json 格式的复盘文件');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const data = JSON.parse(e.target.result);

                if (!data.gameInfo || !data.moves || !data.moveAnalyses) {
                    showUploadError('无效的复盘文件：缺少必要数据');
                    return;
                }

                S.gameId = null;
                S.gameInfo = data.gameInfo;
                S.totalMoves = data.totalMoves || data.moves.length;
                S.moves = data.moves;
                S.moveAnalyses = data.moveAnalyses || [];
                S.problemMoves = data.problemMoves || [];
                S.winrateCurve = data.blackWinrateCurve || [];
                S.whiteWinrateCurve = data.whiteWinrateCurve || [];
                S.result = data;
                S.status = 'reviewed';
                S.currentMove = 0;

                uploadOverlay.classList.add('fade-out');
                appEl.classList.remove('hidden');
                uploadError.classList.add('hidden');

                renderGameInfo();
                moveSlider.max = S.totalMoves;
                moveSlider.value = 0;

                board.size = S.gameInfo.boardSize || 19;
                board.resize();

                chart.setData(S.winrateCurve, S.whiteWinrateCurve, S.problemMoves);
                renderProblemList();
                reviewActions.classList.add('hidden');
                btnSaveReview.classList.remove('hidden');

                goToMove(0);
            } catch (err) {
                showUploadError('解析复盘文件失败: ' + err.message);
            }
        };
        reader.onerror = () => showUploadError('读取文件失败');
        reader.readAsText(file);
    }

    btnSaveReview.addEventListener('click', throttle(exportResults, 2000));

    importInput.addEventListener('change', () => {
        if (importInput.files.length) importResults(importInput.files[0]);
        importInput.value = '';
    });

    // ── New Game ───────────────────────────────────────────

    $('#btn-new-game').addEventListener('click', () => {
        S.gameId = null;
        S.status = 'idle';
        S.result = null;
        S.moveAnalyses = [];
        S.problemMoves = [];
        S.winrateCurve = [];
        S.whiteWinrateCurve = [];
        if (S.ws) { S.ws.close(); S.ws = null; }

        appEl.classList.add('hidden');
        uploadOverlay.classList.remove('fade-out');
        uploadError.classList.add('hidden');
        progressSection.classList.add('hidden');
        reviewActions.classList.remove('hidden');
        btnSaveReview.classList.add('hidden');
        fileInput.value = '';
        importInput.value = '';
    });

    // ── Utils ──────────────────────────────────────────────

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // ── Health check on load ───────────────────────────────

    (async function init() {
        try {
            const h = await api('/api/health');
            if (!h.katagoRunning) {
                console.warn('KataGo 引擎未运行 — 分析功能将不可用');
            }
        } catch (_) {
            console.warn('后端连接失败');
        }
    })();
})();
