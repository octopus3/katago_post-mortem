/**
 * Go board renderer with capture logic.
 * Provides GoLogic (game rules) and GoBoard (canvas rendering).
 */

const GTP_COLS = 'ABCDEFGHJKLMNOPQRST';

function gtpToXY(gtp, size) {
    if (!gtp || gtp.toLowerCase() === 'pass') return null;
    const letter = gtp[0].toUpperCase();
    const col = GTP_COLS.indexOf(letter);
    const row = parseInt(gtp.substring(1), 10);
    if (col < 0 || isNaN(row)) return null;
    return { x: col, y: size - row };
}

function xyToGtp(x, y, size) {
    if (x < 0 || x >= size || y < 0 || y >= size) return '';
    return GTP_COLS[x] + (size - y);
}

// ── Go rules (captures / liberties) ──────────────────────

class GoLogic {
    constructor(size = 19) {
        this.size = size;
        this.board = [];
        this.reset();
    }

    reset() {
        this.board = Array.from({ length: this.size }, () =>
            new Array(this.size).fill(0)
        );
    }

    clone() {
        const c = new GoLogic(this.size);
        for (let y = 0; y < this.size; y++)
            for (let x = 0; x < this.size; x++)
                c.board[y][x] = this.board[y][x];
        return c;
    }

    playMove(x, y, color) {
        if (x < 0 || x >= this.size || y < 0 || y >= this.size) return [];
        if (this.board[y][x] !== 0) return [];
        this.board[y][x] = color;
        const opp = color === 1 ? 2 : 1;
        const captured = [];

        for (const [nx, ny] of this._adj(x, y)) {
            if (this.board[ny][nx] === opp) {
                const grp = this._group(nx, ny);
                if (this._libs(grp) === 0) {
                    for (const [gx, gy] of grp) {
                        this.board[gy][gx] = 0;
                        captured.push({ x: gx, y: gy });
                    }
                }
            }
        }
        const self = this._group(x, y);
        if (this._libs(self) === 0) {
            for (const [gx, gy] of self) this.board[gy][gx] = 0;
        }
        return captured;
    }

    _adj(x, y) {
        const n = [];
        if (x > 0) n.push([x - 1, y]);
        if (x < this.size - 1) n.push([x + 1, y]);
        if (y > 0) n.push([x, y - 1]);
        if (y < this.size - 1) n.push([x, y + 1]);
        return n;
    }

    _group(x, y) {
        const c = this.board[y][x];
        if (!c) return [];
        const vis = new Set();
        const grp = [];
        const stk = [[x, y]];
        while (stk.length) {
            const [cx, cy] = stk.pop();
            const k = cy * this.size + cx;
            if (vis.has(k)) continue;
            vis.add(k);
            if (this.board[cy][cx] !== c) continue;
            grp.push([cx, cy]);
            for (const [nx, ny] of this._adj(cx, cy))
                if (!vis.has(ny * this.size + nx)) stk.push([nx, ny]);
        }
        return grp;
    }

    _libs(grp) {
        const s = new Set();
        for (const [x, y] of grp)
            for (const [nx, ny] of this._adj(x, y))
                if (this.board[ny][nx] === 0) s.add(ny * this.size + nx);
        return s.size;
    }
}

// ── Board renderer ───────────────────────────────────────

class GoBoard {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.size = 19;
        this.padding = 28;
        this._displaySize = 0;

        this.stones = [];
        this.lastMove = null;
        this.showCoords = true;
        this.showNumbers = false;
        this.showBestMove = false;
        this.moveHistory = [];

        this.variationMoves = [];
        this.bestMovePos = null;

        this.onClick = null;

        this._initStones();
        this._bindEvents();
        this.resize();
    }

    _initStones() {
        this.stones = Array.from({ length: this.size }, () =>
            new Array(this.size).fill(0)
        );
    }

    _invalidateBgCache() {
        this._bgCache = null;
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        const dispSize = Math.floor(Math.min(rect.width, rect.height));
        if (dispSize <= 0) return;
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = dispSize * dpr;
        this.canvas.height = dispSize * dpr;
        this.canvas.style.width = dispSize + 'px';
        this.canvas.style.height = dispSize + 'px';
        this._displaySize = dispSize;
        this._dpr = dpr;
        this._invalidateBgCache();
        this.draw();
    }

    get _grid() {
        const pad = this.showCoords ? this.padding : 14;
        return (this._displaySize - 2 * pad) / (this.size - 1);
    }

    get _pad() {
        return this.showCoords ? this.padding : 14;
    }

    _toScreen(bx, by) {
        return {
            x: this._pad + bx * this._grid,
            y: this._pad + by * this._grid,
        };
    }

    _toBoard(sx, sy) {
        const bx = Math.round((sx - this._pad) / this._grid);
        const by = Math.round((sy - this._pad) / this._grid);
        if (bx < 0 || bx >= this.size || by < 0 || by >= this.size) return null;
        return { x: bx, y: by };
    }

    _bindEvents() {
        this.canvas.addEventListener('click', (e) => {
            if (!this.onClick) return;
            const rect = this.canvas.getBoundingClientRect();
            const sx = e.clientX - rect.left;
            const sy = e.clientY - rect.top;
            const pos = this._toBoard(sx, sy);
            if (pos) this.onClick(pos.x, pos.y);
        });

        let resizeTimer;
        const ro = new ResizeObserver(() => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => this.resize(), 60);
        });
        ro.observe(this.canvas.parentElement);
    }

    setPosition(stoneArray, lastMove) {
        this.stones = stoneArray;
        this.lastMove = lastMove;
        this.draw();
    }

    setVariation(moves) {
        this.variationMoves = moves || [];
        this.draw();
    }

    clearVariation() {
        this.variationMoves = [];
        this.draw();
    }

    setBestMove(gtp) {
        this.bestMovePos = gtp ? gtpToXY(gtp, this.size) : null;
        this.draw();
    }

    updateState(stoneArray, lastMove, bestGtp) {
        this.stones = stoneArray;
        this.lastMove = lastMove;
        this.variationMoves = [];
        this.bestMovePos = bestGtp ? gtpToXY(bestGtp, this.size) : null;
        this.draw();
    }

    // ── Main draw ─────────────────────────────────────────

    draw() {
        const ctx = this.ctx;
        const dpr = this._dpr;
        ctx.save();
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this._drawBackground(ctx);
        this._drawGrid(ctx);
        this._drawStarPoints(ctx);
        if (this.showCoords) this._drawCoords(ctx);
        this._drawStones(ctx);
        this._drawLastMove(ctx);
        if (this.showNumbers) this._drawMoveNumbers(ctx);
        if (this.showBestMove && this.bestMovePos) this._drawBestMove(ctx);
        this._drawVariation(ctx);

        ctx.restore();
    }

    _drawBackground(ctx) {
        if (!this._bgCache) {
            const off = document.createElement('canvas');
            off.width = this.canvas.width;
            off.height = this.canvas.height;
            const oc = off.getContext('2d');
            oc.scale(this._dpr, this._dpr);
            oc.fillStyle = '#dcb35c';
            oc.fillRect(0, 0, this._displaySize, this._displaySize);
            oc.strokeStyle = 'rgba(0,0,0,.04)';
            oc.lineWidth = 1;
            for (let i = 0; i < 40; i++) {
                const y = (i / 40) * this._displaySize + ((i * 7.3) % 5);
                oc.beginPath();
                oc.moveTo(0, y);
                oc.lineTo(this._displaySize, y + Math.sin(i * 1.7) * 3);
                oc.stroke();
            }
            this._bgCache = off;
        }
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.drawImage(this._bgCache, 0, 0);
        ctx.restore();
    }

    _drawGrid(ctx) {
        const g = this._grid;
        const p = this._pad;
        const end = p + (this.size - 1) * g;
        ctx.strokeStyle = '#5c4a28';
        ctx.lineWidth = 1;

        for (let i = 0; i < this.size; i++) {
            const v = p + i * g;
            ctx.beginPath();
            ctx.moveTo(v, p);
            ctx.lineTo(v, end);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(p, v);
            ctx.lineTo(end, v);
            ctx.stroke();
        }
    }

    _drawStarPoints(ctx) {
        let pts;
        if (this.size === 19) pts = [[3,3],[3,9],[3,15],[9,3],[9,9],[9,15],[15,3],[15,9],[15,15]];
        else if (this.size === 13) pts = [[3,3],[3,9],[9,3],[9,9],[6,6]];
        else if (this.size === 9) pts = [[2,2],[2,6],[6,2],[6,6],[4,4]];
        else return;

        ctx.fillStyle = '#5c4a28';
        for (const [bx, by] of pts) {
            const s = this._toScreen(bx, by);
            ctx.beginPath();
            ctx.arc(s.x, s.y, Math.max(3, this._grid * 0.12), 0, Math.PI * 2);
            ctx.fill();
        }
    }

    _drawCoords(ctx) {
        const g = this._grid;
        const p = this._pad;
        const end = p + (this.size - 1) * g;
        ctx.fillStyle = '#7c6a3e';
        ctx.font = `${Math.max(9, this._grid * 0.38)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < this.size; i++) {
            const x = p + i * g;
            ctx.fillText(GTP_COLS[i], x, p - 14);
            ctx.fillText(GTP_COLS[i], x, end + 14);
        }
        ctx.textAlign = 'center';
        for (let i = 0; i < this.size; i++) {
            const y = p + i * g;
            const num = this.size - i;
            ctx.fillText(num, p - 16, y);
            ctx.fillText(num, end + 16, y);
        }
    }

    _drawStones(ctx) {
        const r = this._grid * 0.46;
        for (let by = 0; by < this.size; by++) {
            for (let bx = 0; bx < this.size; bx++) {
                const c = this.stones[by] && this.stones[by][bx];
                if (!c) continue;
                const s = this._toScreen(bx, by);
                this._drawStone(ctx, s.x, s.y, r, c);
            }
        }
    }

    _drawStone(ctx, x, y, r, color) {
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,.25)';
        ctx.shadowBlur = r * 0.4;
        ctx.shadowOffsetX = r * 0.1;
        ctx.shadowOffsetY = r * 0.15;

        if (color === 1) {
            const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.05, x, y, r);
            grad.addColorStop(0, '#555');
            grad.addColorStop(1, '#111');
            ctx.fillStyle = grad;
        } else {
            const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.05, x, y, r);
            grad.addColorStop(0, '#fff');
            grad.addColorStop(1, '#ccc');
            ctx.fillStyle = grad;
        }

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        if (color === 2) {
            ctx.strokeStyle = '#aaa';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    _drawLastMove(ctx) {
        if (!this.lastMove) return;
        const s = this._toScreen(this.lastMove.x, this.lastMove.y);
        const r = this._grid * 0.16;
        const stoneColor = this.stones[this.lastMove.y] && this.stones[this.lastMove.y][this.lastMove.x];
        ctx.fillStyle = stoneColor === 1 ? '#e55' : '#d33';
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.fill();
    }

    _drawMoveNumbers(ctx) {
        const r = this._grid * 0.34;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const fontSize = Math.max(8, r * 1.2);
        ctx.font = `bold ${fontSize}px sans-serif`;

        for (let i = 0; i < this.moveHistory.length; i++) {
            const m = this.moveHistory[i];
            if (!m || m.isPass) continue;
            const pos = gtpToXY(m.gtpCoord, this.size);
            if (!pos) continue;
            if (!this.stones[pos.y] || !this.stones[pos.y][pos.x]) continue;
            const s = this._toScreen(pos.x, pos.y);
            const stoneColor = this.stones[pos.y][pos.x];
            ctx.fillStyle = stoneColor === 1 ? '#fff' : '#222';
            ctx.fillText(m.moveNumber, s.x, s.y + 0.5);
        }
    }

    _drawBestMove(ctx) {
        const bp = this.bestMovePos;
        if (!bp) return;
        if (this.stones[bp.y] && this.stones[bp.y][bp.x]) return;
        const s = this._toScreen(bp.x, bp.y);
        const r = this._grid * 0.28;
        ctx.strokeStyle = '#16a34a';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.stroke();

        ctx.fillStyle = 'rgba(22,163,74,.15)';
        ctx.beginPath();
        ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
        ctx.fill();
    }

    _drawVariation(ctx) {
        if (!this.variationMoves.length) return;
        const r = this._grid * 0.42;
        const fontSize = Math.max(9, r * 1.1);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < this.variationMoves.length; i++) {
            const vm = this.variationMoves[i];
            const pos = gtpToXY(vm.gtp, this.size);
            if (!pos) continue;
            const s = this._toScreen(pos.x, pos.y);

            ctx.globalAlpha = 0.7;
            if (vm.color === 'B') {
                ctx.fillStyle = '#333';
            } else {
                ctx.fillStyle = '#eee';
            }
            ctx.beginPath();
            ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
            ctx.fill();

            if (vm.color === 'W') {
                ctx.strokeStyle = '#999';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
                ctx.stroke();
            }

            ctx.globalAlpha = 1;
            ctx.fillStyle = vm.color === 'B' ? '#fff' : '#222';
            ctx.font = `bold ${fontSize}px sans-serif`;
            ctx.fillText(i + 1, s.x, s.y + 0.5);
        }
    }
}
