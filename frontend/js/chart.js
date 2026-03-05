/**
 * Winrate chart wrapper built on Chart.js.
 * Supports black / white / both view modes.
 */

class WinrateChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.chart = null;
        this.onMoveClick = null;
        this._currentMove = 0;
        this._totalMoves = 0;
        this._blackCurve = [];
        this._whiteCurve = [];
        this._problemMoves = [];
        this._viewMode = 'both'; // 'black' | 'white' | 'both'
        this._init();
    }

    _init() {
        const self = this;

        const currentLinePlugin = {
            id: 'currentMoveLine',
            afterDraw(chart) {
                if (!self._totalMoves) return;
                const { ctx, chartArea, scales } = chart;
                const xPixel = scales.x.getPixelForValue(self._currentMove);
                if (xPixel < chartArea.left || xPixel > chartArea.right) return;

                ctx.save();
                ctx.strokeStyle = 'rgba(37,99,235,.6)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.beginPath();
                ctx.moveTo(xPixel, chartArea.top);
                ctx.lineTo(xPixel, chartArea.bottom);
                ctx.stroke();
                ctx.restore();
            },
        };

        const halfLinePlugin = {
            id: 'halfLine',
            beforeDraw(chart) {
                const { ctx, chartArea, scales } = chart;
                const yPixel = scales.y.getPixelForValue(50);
                ctx.save();
                ctx.strokeStyle = 'rgba(0,0,0,.12)';
                ctx.lineWidth = 1;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(chartArea.left, yPixel);
                ctx.lineTo(chartArea.right, yPixel);
                ctx.stroke();
                ctx.restore();
            },
        };

        this.chart = new Chart(this.canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '黑方胜率',
                        data: [],
                        borderColor: '#333',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.15,
                    },
                    {
                        label: '白方胜率',
                        data: [],
                        borderColor: '#aaa',
                        borderWidth: 2,
                        borderDash: [6, 3],
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: false,
                        tension: 0.15,
                    },
                    {
                        label: '问题手',
                        data: [],
                        borderWidth: 0,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        pointBackgroundColor: [],
                        pointBorderColor: '#fff',
                        pointBorderWidth: 1.5,
                        showLine: false,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: { display: false },
                        ticks: {
                            maxTicksLimit: 12,
                            font: { size: 10 },
                            color: '#94a3b8',
                        },
                        grid: { display: false },
                    },
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 25,
                            callback: v => v + '%',
                            font: { size: 10 },
                            color: '#94a3b8',
                        },
                        grid: { color: 'rgba(0,0,0,.05)' },
                    },
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: items => `第 ${items[0].label} 手`,
                            label: item => {
                                const names = ['黑方胜率', '白方胜率', '问题手'];
                                return `${names[item.datasetIndex] || ''}: ${item.parsed.y.toFixed(1)}%`;
                            },
                        },
                    },
                },
                onClick(event, elements) {
                    if (!elements.length || !self.onMoveClick) return;
                    self.onMoveClick(elements[0].index);
                },
            },
            plugins: [currentLinePlugin, halfLinePlugin],
        });
    }

    setData(blackCurve, whiteCurve, problemMoves) {
        if (!blackCurve || !blackCurve.length) return;
        this._blackCurve = blackCurve;
        this._whiteCurve = whiteCurve || blackCurve.map(v => 1 - v);
        this._problemMoves = problemMoves || [];
        this._totalMoves = blackCurve.length - 1;
        this._applyView();
    }

    setViewMode(mode) {
        this._viewMode = mode;
        this._applyView();
    }

    _applyView() {
        const bc = this._blackCurve;
        const wc = this._whiteCurve;
        if (!bc.length) return;

        const labels = bc.map((_, i) => i);
        const blackPct = bc.map(v => +(v * 100).toFixed(2));
        const whitePct = wc.map(v => +(v * 100).toFixed(2));

        const showBlack = this._viewMode === 'black' || this._viewMode === 'both';
        const showWhite = this._viewMode === 'white' || this._viewMode === 'both';

        const activeCurve = this._viewMode === 'white' ? whitePct : blackPct;

        const problemData = new Array(bc.length).fill(null);
        const problemColors = new Array(bc.length).fill('transparent');
        const sevColorMap = { minor: '#f59e0b', questionable: '#ea580c', bad: '#dc2626' };

        for (const pm of this._problemMoves) {
            const idx = pm.moveNumber;
            if (idx >= 0 && idx < bc.length) {
                problemData[idx] = activeCurve[idx];
                problemColors[idx] = sevColorMap[pm.severity] || '#dc2626';
            }
        }

        this.chart.data.labels = labels;

        // Black line
        this.chart.data.datasets[0].data = showBlack ? blackPct : [];
        this.chart.data.datasets[0].borderWidth = this._viewMode === 'both' ? 2 : 2.5;
        this.chart.data.datasets[0].fill = this._viewMode === 'black'
            ? { target: { value: 50 }, above: 'rgba(30,30,30,.12)', below: 'rgba(200,200,200,.25)' }
            : false;

        // White line
        this.chart.data.datasets[1].data = showWhite ? whitePct : [];
        this.chart.data.datasets[1].borderWidth = this._viewMode === 'both' ? 2 : 2.5;
        this.chart.data.datasets[1].fill = this._viewMode === 'white'
            ? { target: { value: 50 }, above: 'rgba(200,200,200,.25)', below: 'rgba(30,30,30,.12)' }
            : false;

        // Problem dots
        this.chart.data.datasets[2].data = problemData;
        this.chart.data.datasets[2].pointBackgroundColor = problemColors;

        this.chart.update();
    }

    highlightMove(n) {
        this._currentMove = n;
        this.chart.update('none');
    }

    destroy() {
        if (this.chart) this.chart.destroy();
    }
}
