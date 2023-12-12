var data = [];

let n = data.length;
for(let i = 0; i < n; i++) {
data.push({
  x: Object.values(table.x),
  open: Object.values(table.open),
  high: Object.values(table.high),
  low: Object.values(table.low),
  close: Object.values(table.close),
  xaxis: `x1`,
  yaxis: `y1`,
  type: 'candlestick'
}, {
  x: Object.values(table.x),
  y: Object.values(table["corr_ta_rsi_7"]),
  xaxis: `x2`,
  yaxis: `y1`,
  type: "scatter"
});
}

// Layout options for the plot
var layout = {
title: 'My plot',
width: 800,
height: 800,
grid: {
  rows: 3,
  columns: n,
  pattern: "independent",
}
};

Plotly.newPlot('myPlot1', data, layout);