/**
 * Winrate chart wrapper built on Chart.js.
 */

class WinrateChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.chart = null;
        this.onMoveClick = null;
        this._currentMove = 0;
        this._totalMoves = 0;
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
                        fill: {
                            target: { value: 50 },
                            above: 'rgba(30,30,30,.12)',
                            below: 'rgba(200,200,200,.25)',
                        },
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
                                if (item.datasetIndex === 0) return `黑方胜率: ${item.parsed.y.toFixed(1)}%`;
                                return `问题手: ${item.parsed.y.toFixed(1)}%`;
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

    setData(curve, problemMoves) {
        if (!curve || !curve.length) return;
        this._totalMoves = curve.length - 1;

        const labels = curve.map((_, i) => i);
        const pctData = curve.map(v => +(v * 100).toFixed(2));

        const problemData = new Array(curve.length).fill(null);
        const problemColors = new Array(curve.length).fill('transparent');

        const sevColorMap = {
            minor: '#f59e0b',
            questionable: '#ea580c',
            bad: '#dc2626',
        };

        if (problemMoves) {
            for (const pm of problemMoves) {
                const idx = pm.moveNumber;
                if (idx >= 0 && idx < curve.length) {
                    problemData[idx] = pctData[idx];
                    problemColors[idx] = sevColorMap[pm.severity] || '#dc2626';
                }
            }
        }

        this.chart.data.labels = labels;
        this.chart.data.datasets[0].data = pctData;
        this.chart.data.datasets[1].data = problemData;
        this.chart.data.datasets[1].pointBackgroundColor = problemColors;
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
