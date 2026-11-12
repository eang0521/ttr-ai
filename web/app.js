'use strict';

// The drawing is W wide; its height and scale follow the board on show (drawBoard). The USA
// board uses the official board's proportions (see board_layout.js); other boards bring
// their own size, city positions and landmarks from the server.
const W = 1400;
let BOARD_SCALE = W / BOARD_SIZE[0], H = Math.round(BOARD_SIZE[1] * BOARD_SCALE);
const CARD_TYPES = ['purple', 'white', 'blue', 'yellow', 'orange', 'black', 'red', 'green', 'locomotive'];
const LOCO = 8;
const BASE_ROUTE_FILL = {
  purple: '#b05fb5', white: '#f3f1ea', blue: '#2f74c0', yellow: '#f0c93a', orange: '#ec8a2c',
  black: '#2a2a2a', red: '#d13b35', green: '#3d9a4a', gray: '#a5a39c',
};
let ROUTE_FILL = { ...BASE_ROUTE_FILL };
const PLAYER_FILL = { red: '#b8312f', blue: '#2457a8', green: '#23854a', yellow: '#e2b417', purple: '#6a3d9a' };
const ROUTE_POINTS = { 1: 1, 2: 2, 3: 4, 4: 7, 5: 10, 6: 15 };
const CITY_R = 9;
const CAR_H = 11;
let CAR_PITCH = 48;  // length of every train spot, including the gap to the next one (per board)
const DOUBLE_GAP = 6.5;

// Label placement [dx, dy, anchor], roughly where the official board prints each name.
const LABELS = {
  'Vancouver': [-4, -15, 'middle'], 'Seattle': [-13, 4, 'end'], 'Portland': [12, 18, 'start'],
  'Calgary': [0, -15, 'middle'], 'Winnipeg': [4, -15, 'middle'], 'Helena': [-6, 24, 'middle'],
  'Duluth': [-12, -10, 'end'], 'Sault St. Marie': [4, -15, 'middle'], 'Toronto': [13, 4, 'start'],
  'Montreal': [-6, -15, 'middle'], 'Boston': [-12, 18, 'end'], 'New York': [13, 16, 'start'],
  'Pittsburgh': [13, -6, 'start'], 'Washington': [13, 16, 'start'], 'Raleigh': [13, 4, 'start'],
  'Charleston': [13, 5, 'start'], 'Chicago': [13, 16, 'start'], 'Saint Louis': [13, -8, 'start'],
  'Nashville': [13, -6, 'start'], 'Atlanta': [13, -6, 'start'], 'Miami': [13, 16, 'start'],
  'New Orleans': [0, 24, 'middle'], 'Houston': [10, 22, 'start'], 'Dallas': [13, 14, 'start'],
  'Little Rock': [13, 16, 'start'], 'Oklahoma City': [10, -14, 'start'], 'Kansas City': [6, 24, 'middle'],
  'Omaha': [13, 5, 'start'], 'Denver': [13, 16, 'start'], 'Santa Fe': [-13, 4, 'end'],
  'El Paso': [0, 24, 'middle'], 'Phoenix': [-12, 18, 'end'], 'Las Vegas': [10, 20, 'start'],
  'Los Angeles': [-8, 22, 'middle'], 'San Francisco': [0, 24, 'start'], 'Salt Lake City': [-13, -10, 'end'],
};

const $ = s => document.querySelector(s);
const api = {
  get: u => fetch(u).then(r => r.json()),
  post: (u, b) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) }).then(r => r.json()),
};

let MAP = null;
let state = null, viewer = null, lastHuman = null, revealedFor = null;
let botTimer = null, busy = false, modalKind = null, ticketSelection = null;
let payColor = null;  // card type picked in the hand before clicking a route
let reviewTurn = null, liveTurns = 0, botsPaused = false;  // reviewTurn: null = live game
let cityPos = {};

// ------------------------------------------------------------------ board

const isUSA = () => MAP.layout === 'geo';

/** Card color name as this board shows it (e.g. "pink" for purple on some boards). */
function colorLabel(c) {
  const style = MAP && MAP.color_style && MAP.color_style[c];
  return style ? style[0].toLowerCase() : c;
}

/** Load the board a game is played on, restyle the cards and redraw the map. */
async function loadBoard(key) {
  MAP = await api.get(`/api/map?board=${encodeURIComponent(key || 'usa')}`);
  document.body.classList.toggle('local-board', !isUSA());
  ROUTE_FILL = { ...BASE_ROUTE_FILL };
  const root = document.documentElement.style;
  for (const c of CARD_TYPES.slice(0, 8)) {
    const style = MAP.color_style[c];
    ROUTE_FILL[c] = style ? style[1] : BASE_ROUTE_FILL[c];
    root.setProperty(`--${c}`, ROUTE_FILL[c]);
  }
  await drawBoard();
}

async function drawBoard() {
  const usa = isUSA();
  const size = usa ? BOARD_SIZE : MAP.size;
  BOARD_SCALE = W / size[0];
  H = Math.round(size[1] * BOARD_SCALE);
  CAR_PITCH = usa ? 48 : (MAP.ui.car_pitch || 48);
  cityPos = {};
  const svg = d3.select('#board').attr('viewBox', `0 0 ${W} ${H}`);
  svg.selectAll('*').remove();
  if (usa) {
    for (const [name, [x, y]] of Object.entries(BOARD_CITIES)) cityPos[name] = [x * BOARD_SCALE, y * BOARD_SCALE];
  } else {
    for (const c of MAP.cities) cityPos[c.name] = [c.x * BOARD_SCALE, c.y * BOARD_SCALE];
  }

  const defs = svg.append('defs');
  defs.append('clipPath').attr('id', 'boardClip').append('rect').attr('width', W).attr('height', H);
  const shadow = defs.append('filter').attr('id', 'trainShadow').attr('x', '-20%').attr('y', '-40%')
    .attr('width', '140%').attr('height', '180%');
  shadow.append('feDropShadow').attr('dx', 1.6).attr('dy', 2.4).attr('stdDeviation', 1.3)
    .attr('flood-color', '#1a0d04').attr('flood-opacity', 0.6);
  for (const [name, hex] of Object.entries(PLAYER_FILL)) {
    // Light from above: a lit roof edge, the body color, then a shaded lower side.
    const c = d3.color(hex);
    const grad = defs.append('linearGradient').attr('id', `car-${name}`).attr('x1', 0).attr('y1', 0).attr('x2', 0).attr('y2', 1);
    [[0, d3.interpolateRgb(c, '#fff')(0.55)], [0.28, d3.interpolateRgb(c, '#fff')(0.15)], [0.55, c.formatHex()],
     [1, d3.interpolateRgb(c, '#000')(0.55)]].forEach(([o, col]) =>
      grad.append('stop').attr('offset', o).attr('stop-color', col));
  }
  const paper = defs.append('filter').attr('id', 'paper').attr('x', 0).attr('y', 0).attr('width', 1).attr('height', 1);
  paper.append('feTurbulence').attr('type', 'fractalNoise').attr('baseFrequency', 0.9).attr('numOctaves', 3).attr('seed', 4);
  paper.append('feColorMatrix').attr('values', '0 0 0 0 0.35  0 0 0 0 0.24  0 0 0 0 0.1  0 0 0 0.16 0');
  paper.append('feComposite').attr('in2', 'SourceGraphic').attr('operator', 'in');
  const vignette = defs.append('radialGradient').attr('id', 'vignette').attr('r', 0.75);
  vignette.append('stop').attr('offset', '60%').attr('stop-color', '#3b2412').attr('stop-opacity', 0);
  vignette.append('stop').attr('offset', '100%').attr('stop-color', '#3b2412').attr('stop-opacity', 0.45);
  const rainbow = defs.append('linearGradient').attr('id', 'rainbow');
  ['#d13b35', '#ec8a2c', '#f0c93a', '#3d9a4a', '#2f74c0', '#b05fb5'].forEach((c, i, a) =>
    rainbow.append('stop').attr('offset', `${(100 * i) / (a.length - 1)}%`).attr('stop-color', c));

  if (usa) await drawUSABackground(svg);
  else drawLocalBackground(svg);
  svg.append('rect').attr('width', W).attr('height', H).attr('filter', 'url(#paper)').attr('pointer-events', 'none');
  svg.append('rect').attr('width', W).attr('height', H).attr('fill', 'url(#vignette)').attr('pointer-events', 'none');
  drawCompass(svg);

  // Routes.
  const routeLayer = svg.append('g').attr('id', 'routes');
  const routes = routeLayer.selectAll('g.route').data(MAP.routes).join('g')
    .attr('class', 'route').attr('data-id', r => r.id);
  routes.each(function (r) {
    const g = d3.select(this);
    r.geom = routeGeometry(r);
    g.append('path').attr('class', 'hit').attr('d', r.geom.hitPath);
    g.selectAll('rect.route-slot').data(r.geom.cars).join('rect')
      .attr('class', `route-slot ${r.color === 'gray' ? 'gray' : ''}`)
      .attr('x', c => -c.w / 2).attr('y', -CAR_H / 2).attr('width', c => c.w).attr('height', CAR_H)
      .attr('rx', 1.5).attr('fill', ROUTE_FILL[r.color])
      .attr('transform', c => `translate(${c.x},${c.y}) rotate(${c.angle})`);
    g.append('g').attr('class', 'claimed');
  });
  routes.on('click', (e, r) => onRouteClick(r))
    .on('mousemove', (e, r) => showTooltip(e, routeTooltip(r)))
    .on('mouseleave', hideTooltip);

  // Cities on top.
  const cityLayer = svg.append('g').attr('id', 'cities');
  const cg = cityLayer.selectAll('g.city').data(MAP.cities).join('g')
    .attr('class', 'city').attr('data-name', c => c.name)
    .attr('transform', c => `translate(${cityPos[c.name]})`);
  cg.append('circle').attr('class', 'halo').attr('r', CITY_R + 5);
  cg.append('circle').attr('class', 'dot').attr('r', CITY_R);
  const labels = usa ? LABELS : (MAP.ui.labels || {});
  cityLayer.selectAll('text').data(MAP.cities).join('text')
    .attr('class', 'city-label').attr('data-name', c => c.name)
    .attr('x', c => cityPos[c.name][0] + (labels[c.name] || [12])[0])
    .attr('y', c => cityPos[c.name][1] + (labels[c.name] || [12, 4])[1])
    .attr('text-anchor', c => (labels[c.name] || [0, 0, 'start'])[2])
    .text(c => c.name);
}

/** USA: sea, then land warped so real coastlines and borders sit where the board draws them. */
async function drawUSABackground(svg) {
  const [world, us] = await Promise.all([
    api.get('vendor/countries-50m.json'), api.get('vendor/states-10m.json'),
  ]);
  const warp = boardWarp();
  const path = d3.geoPath(d3.geoTransform({ point(lon, lat) { this.stream.point(...warp(lon, lat)); } }));
  svg.append('rect').attr('width', W).attr('height', H).attr('fill', '#9fbfc4');
  const geo = svg.append('g').attr('clip-path', 'url(#boardClip)');
  const NEIGHBORS = new Set(['124', '484', '192', '044']);  // Canada, Mexico, Cuba, Bahamas
  const countries = topojson.feature(world, world.objects.countries).features;
  geo.append('g').selectAll('path').data(countries.filter(f => NEIGHBORS.has(f.id))).join('path')
    .attr('d', path).attr('fill', '#e3d3a6').attr('stroke', '#8f7a52').attr('stroke-width', 0.8);
  const states = topojson.feature(us, us.objects.states).features
    .filter(f => !['02', '15', '72', '78', '60', '66', '69'].includes(f.id));  // no AK, HI, territories
  geo.append('g').selectAll('path').data(states).join('path')
    .attr('d', path).attr('fill', '#f1e2b9');
  geo.append('path').datum(topojson.mesh(us, us.objects.states, (a, b) => a !== b)).attr('d', path)
    .attr('fill', 'none').attr('stroke', '#b39b6a').attr('stroke-width', 0.8).attr('stroke-dasharray', '3 2');
}

/** Other boards: a parchment map with their landmarks (parks, freeways) and title. */
function drawLocalBackground(svg) {
  const decor = MAP.ui.decor || {};
  const at = ([x, y]) => [x * BOARD_SCALE, y * BOARD_SCALE];
  svg.append('rect').attr('width', W).attr('height', H).attr('fill', '#f1e2b9');
  const geo = svg.append('g').attr('clip-path', 'url(#boardClip)');
  const smooth = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.5));
  for (const park of decor.parks || []) {
    geo.append('path').attr('d', smooth(park.points.map(at))).attr('fill', '#d6dcae')
      .attr('stroke', '#a9b07a').attr('stroke-width', 1.2).attr('stroke-dasharray', '4 3').attr('opacity', 0.85);
    geo.append('text').attr('class', 'map-note').attr('text-anchor', 'middle')
      .attr('x', at(park.label)[0]).attr('y', at(park.label)[1]).text(park.name);
  }
  const road = d3.line().curve(d3.curveCatmullRom.alpha(0.5));
  for (const r of decor.roads || []) {
    const d = road(r.points.map(at));
    geo.append('path').attr('d', d).attr('fill', 'none').attr('stroke', '#c7b183').attr('stroke-width', 7)
      .attr('stroke-linecap', 'round').attr('opacity', 0.7);
    geo.append('path').attr('d', d).attr('fill', 'none').attr('stroke', '#f6ecd0').attr('stroke-width', 2.5)
      .attr('stroke-linecap', 'round').attr('opacity', 0.8);
  }
  for (const r of decor.roads || []) {
    const [x, y] = at(r.label);
    const shield = geo.append('g').attr('class', 'road-shield').attr('transform', `translate(${x},${y})`);
    shield.append('rect').attr('x', -15).attr('y', -10).attr('width', 30).attr('height', 20).attr('rx', 5);
    shield.append('text').attr('text-anchor', 'middle').attr('y', 5).text(r.name);
  }
  svg.append('text').attr('class', 'board-title').attr('x', 24).attr('y', 44).text(MAP.title);
}

/**
 * Rubber-sheet warp from longitude/latitude to board coordinates: a thin-plate spline
 * through all 36 cities (real position -> board position), so the real map bends the
 * same way the board's artwork does.
 */
function boardWarp() {
  const proj = d3.geoConicConformal().parallels([33, 45]).rotate([96, 0]).scale(1000).translate([0, 0]);
  const pts = MAP.cities.map(c => ({ src: proj([c.lon, c.lat]), dst: cityPos[c.name] }));
  const n = pts.length;
  const U = r2 => (r2 === 0 ? 0 : r2 * Math.log(r2) / 2);  // r^2 log r, from r^2
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const lambda = 0.02 * d3.mean(pts, p => d3.mean(pts, q => U(d2(p.src, q.src))));
  // Solve [[K + lambda*I, P], [P^T, 0]] [w; a] = [v; 0] for x and y together.
  const m = n + 3;
  const A = Array.from({ length: m }, () => new Float64Array(m + 2));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i][j] = U(d2(pts[i].src, pts[j].src)) + (i === j ? lambda : 0);
    A[i][n] = A[n][i] = 1;
    A[i][n + 1] = A[n + 1][i] = pts[i].src[0];
    A[i][n + 2] = A[n + 2][i] = pts[i].src[1];
    A[i][m] = pts[i].dst[0];
    A[i][m + 1] = pts[i].dst[1];
  }
  for (let c = 0; c < m; c++) {  // Gaussian elimination with partial pivoting
    let p = c;
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < m; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < m + 2; k++) A[r][k] -= f * A[c][k];
    }
  }
  const wx = A.map((row, i) => row[m] / row[i]), wy = A.map((row, i) => row[m + 1] / row[i]);
  return (lon, lat) => {
    const s = proj([lon, lat]);
    let x = wx[n] + wx[n + 1] * s[0] + wx[n + 2] * s[1];
    let y = wy[n] + wy[n + 1] * s[0] + wy[n + 2] * s[1];
    for (let i = 0; i < n; i++) {
      const u = U(d2(s, pts[i].src));
      x += wx[i] * u;
      y += wy[i] * u;
    }
    return [x, y];
  };
}

function drawCompass(svg) {
  const g = svg.append('g').attr('transform', `translate(95,${H - 230})`).attr('opacity', 0.75);
  g.append('circle').attr('r', 46).attr('fill', 'none').attr('stroke', '#6b4a2a').attr('stroke-width', 1.5);
  g.append('circle').attr('r', 40).attr('fill', 'none').attr('stroke', '#6b4a2a').attr('stroke-width', 0.8);
  for (let i = 0; i < 4; i++) {
    g.append('path').attr('d', 'M0,-52 L8,0 L0,8 L-8,0 Z').attr('fill', i % 2 ? '#e3d3a6' : '#6b3f1f')
      .attr('stroke', '#3b2412').attr('transform', `rotate(${i * 90})`);
  }
  g.append('text').attr('y', -58).attr('text-anchor', 'middle').attr('class', 'city-label').text('N');
}

/** [bend, side, shift, via, ends] for a route on a board without a hand-made USA-style
 * layout. The board's SHAPES may give a route's bend (board units the curve bows to the
 * left of the line from its first city), a zigzag `via` [o1, o2] instead, and `ends`
 * [e1, e2]: extra room (screen pixels) between each end city and the first spot.
 * Otherwise a route is curved just enough for its spots, bowing away from other cities. */
function autoShape(r) {
  const side = r.pair === null ? 0 : (r.id < r.pair ? -1 : 1);
  let drawn = (MAP.ui.shapes || {})[`${r.a}|${r.b}`];
  if (typeof drawn === 'number') drawn = { bend: drawn };
  const bend = drawn && drawn.bend !== undefined ? drawn.bend : (drawn && drawn.via ? 0 : fitBend(r));
  return [bend, side, 0, drawn ? drawn.via : undefined, drawn ? drawn.ends : undefined];
}

/** The bend (board units) that makes a route just long enough for its spots, if needed. */
function fitBend(r) {
  const A = cityPos[r.a], B = cityPos[r.b];
  const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy);
  const nx = -dy / len, ny = dx / len;
  const need = 2 * (CITY_R + 3) + CAR_PITCH * r.length + 4;
  const curveLen = h => {  // length of the quadratic curve with its control point h off the middle
    let total = 0, prev = A;
    for (let i = 1; i <= 40; i++) {
      const t = i / 40, u = 1 - t;
      const p = [u * u * A[0] + 2 * u * t * ((A[0] + B[0]) / 2 + nx * h) + t * t * B[0],
                 u * u * A[1] + 2 * u * t * ((A[1] + B[1]) / 2 + ny * h) + t * t * B[1]];
      total += Math.hypot(p[0] - prev[0], p[1] - prev[1]);
      prev = p;
    }
    return total;
  };
  if (len >= need) return 0;
  let lo = 0, hi = len;
  for (let i = 0; i < 30; i++) { const m = (lo + hi) / 2; if (curveLen(m) < need) lo = m; else hi = m; }
  // Bow toward whichever side keeps the curve's middle further from other cities.
  const clearance = sign => {
    const m = [(A[0] + B[0]) / 2 + nx * hi * sign / 2, (A[1] + B[1]) / 2 + ny * hi * sign / 2];
    return d3.min(MAP.cities.filter(c => c.name !== r.a && c.name !== r.b),
                  c => Math.hypot(cityPos[c.name][0] - m[0], cityPos[c.name][1] - m[1]));
  };
  const sign = clearance(1) >= clearance(-1) ? 1 : -1;
  return sign * hi / BOARD_SCALE;
}

function routeGeometry(r) {
  const A = cityPos[r.a], B = cityPos[r.b];
  const dx = B[0] - A[0], dy = B[1] - A[1], len = Math.hypot(dx, dy);
  const nx = -dy / len, ny = dx / len;
  const usa = isUSA();
  const [boardBend, side, shift, via, ends] = usa ? BOARD_ROUTES[r.id] : autoShape(r);
  // The route's center line; each half of a double route runs parallel to it, so the two
  // halves' spots line up side by side.
  const mid = [];
  if (via) {
    // A gentle zigzag through points 1/3 and 2/3 of the way along, offset sideways
    // (Catmull-Rom spline), like routes on the real board that wiggle to fit their space.
    const P = [A, ...via.map((o, k) => [A[0] + dx * (k + 1) / 3 + nx * o * BOARD_SCALE,
                                         A[1] + dy * (k + 1) / 3 + ny * o * BOARD_SCALE]), B];
    const Q = [P[0], ...P, P[3]];
    for (let seg = 0; seg < 3; seg++) {
      const [p0, p1, p2, p3] = Q.slice(seg, seg + 4);
      for (let i = seg ? 1 : 0; i <= 27; i++) {
        const t = i / 27, t2 = t * t, t3 = t2 * t;
        mid.push([0, 1].map(k => 0.5 * (2 * p1[k] + (p2[k] - p0[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
          + (3 * p1[k] - p0[k] - 3 * p2[k] + p3[k]) * t3)));
      }
    }
  } else {
    const bend = boardBend * BOARD_SCALE;
    const p1 = [(A[0] + B[0]) / 2 + nx * bend, (A[1] + B[1]) / 2 + ny * bend];
    for (let i = 0; i <= 80; i++) {
      const t = i / 80, u = 1 - t;
      mid.push([u * u * A[0] + 2 * u * t * p1[0] + t * t * B[0], u * u * A[1] + 2 * u * t * p1[1] + t * t * B[1]]);
    }
  }
  const off = side * DOUBLE_GAP;
  const pts = mid.map((p, i) => {
    const a = mid[Math.max(0, i - 1)], b = mid[Math.min(mid.length - 1, i + 1)];
    const tl = Math.hypot(b[0] - a[0], b[1] - a[1]);
    return [p[0] - (b[1] - a[1]) / tl * off, p[1] + (b[0] - a[0]) / tl * off];
  });
  const cum = [0];
  for (let i = 1; i < mid.length; i++) cum.push(cum[i - 1] + Math.hypot(mid[i][0] - mid[i - 1][0], mid[i][1] - mid[i - 1][1]));
  const total = cum[cum.length - 1];
  // Every spot is the same size; any spare length is left at the ends (see board_layout.js).
  const trim = CITY_R + 3;
  const slack = Math.max(0, total - 2 * trim - CAR_PITCH * r.length);
  const first = trim + slack / 2 + shift * slack / 2;
  const at = s => {
    let i = d3.bisectLeft(cum, s);
    i = Math.max(1, Math.min(pts.length - 1, i));
    const t = (s - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
    const x = pts[i - 1][0] + t * (pts[i][0] - pts[i - 1][0]);
    const y = pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]);
    let angle = Math.atan2(mid[i][1] - mid[i - 1][1], mid[i][0] - mid[i - 1][0]) * 180 / Math.PI;
    if (angle > 90) angle -= 180;  // keep each car's lit side facing up the screen
    if (angle < -90) angle += 180;
    return { x, y, angle };
  };
  const cars = [];
  // USA: spots packed together, spare length at the ends. Other boards: spots spread
  // evenly along the route, like the dashes on a hand-drawn board.
  // Where there's room, keep spots a little further from the cities so routes leaving a
  // city at a narrow angle don't touch (the board's SHAPES can set each end's gap).
  let [e1, e2] = ends || [Math.min(slack / 2, CAR_PITCH * 0.6), Math.min(slack / 2, CAR_PITCH * 0.6)];
  if (e1 + e2 > slack) { const f = slack / (e1 + e2 || 1); e1 *= f; e2 *= f; }  // spots never shrink
  const pitch = usa ? CAR_PITCH : (total - 2 * trim - e1 - e2) / r.length;
  const start = usa ? first : trim + e1;
  for (let k = 0; k < r.length; k++) {
    cars.push({ ...at(start + pitch * (k + 0.5)), w: CAR_PITCH - 3 });
  }
  return { cars, hitPath: d3.line()(pts) };
}

function routeTooltip(r) {
  let owner = '';
  if (state && state.route_owner[r.id] !== -1) owner = ` · claimed by ${state.players[state.route_owner[r.id]].name}`;
  return `<b>${r.a} – ${r.b}</b><br>${r.length} ${r.color === 'gray' ? 'gray (any one color)' : colorLabel(r.color)} · ${ROUTE_POINTS[r.length]} pts${owner}`;
}

function showTooltip(e, html) {
  const tip = $('#tooltip');
  const box = $('#boardWrap').getBoundingClientRect();
  tip.innerHTML = html;
  tip.classList.remove('hidden');
  tip.style.left = `${e.clientX - box.left + 14}px`;
  tip.style.top = `${e.clientY - box.top + 14}px`;
}
function hideTooltip() { $('#tooltip').classList.add('hidden'); }

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.add('hidden'), 2800);
}

// ------------------------------------------------------------- game flow

async function refresh() {
  let s = await api.get('/api/state' + (viewer !== null ? `?viewer=${viewer}` : ''));
  if (s.no_game) { state = null; render(); showNewGame(); return; }
  if ((s.board || 'usa') !== MAP.board) await loadBoard(s.board);
  const humans = s.players.filter(p => p.type === 'human').map(p => p.index);
  let want;
  if (s.phase !== 'game_over' && humans.includes(s.current)) want = s.current;
  else want = lastHuman !== null && humans.includes(lastHuman) ? lastHuman : (humans.length ? humans[0] : null);
  if (want !== viewer) {
    viewer = want;
    s = await api.get('/api/state' + (viewer !== null ? `?viewer=${viewer}` : ''));
  }
  if (viewer !== null) lastHuman = viewer;
  liveTurns = s.turns;
  if (reviewTurn !== null && (!s.replayable || reviewTurn >= s.turns)) reviewTurn = null;
  if (reviewTurn !== null) {
    // In pass-and-play, review shows only public information (nobody's hand).
    const rv = humans.length > 1 ? null : viewer;
    s = await api.get(`/api/state?turn=${reviewTurn}` + (rv !== null ? `&viewer=${rv}` : ''));
  }
  state = s;
  document.body.classList.toggle('reviewing', !!state.review);
  render();
  scheduleBot();
}

function humanCount() { return state ? state.players.filter(p => p.type === 'human').length : 0; }
function myTurn() { return state && !state.review && state.legal && viewer === state.current && !coverActive(); }
function coverActive() {
  return state && !state.review && humanCount() > 1 && state.phase !== 'game_over' &&
    state.players[state.current].type === 'human' && revealedFor !== state.current;
}

// Pause before each bot move, in ms: [normal move, quick follow-up (second draw, opening tickets)].
const BOT_SPEEDS = {
  slow:    { delays: [1600, 700], hint: 'About 1.5 seconds per bot move.' },
  normal:  { delays: [750, 350],  hint: 'About ¾ second per bot move.' },
  fast:    { delays: [300, 120],  hint: 'About ¼ second per bot move.' },
  instant: { delays: [0, 0],      hint: 'Bots move as fast as they can.' },
};
let botSpeed = 'normal';

function scheduleBot() {
  clearTimeout(botTimer);
  if (!state || state.review || botsPaused || !state.current_is_bot || state.phase === 'game_over') return;
  const quick = state.phase === 'setup_tickets' || state.phase === 'second_draw';
  const [normal, short] = BOT_SPEEDS[botSpeed].delays;
  botTimer = setTimeout(async () => {
    await api.post('/api/bot', {});
    refresh();
  }, quick ? short : normal);
}

function setBotSpeed(speed) {
  if (!BOT_SPEEDS[speed]) speed = 'normal';
  botSpeed = speed;
  try { localStorage.setItem('ttr-bot-speed', speed); } catch (e) { /* ignore */ }
  document.querySelectorAll('#botSpeed button').forEach(b =>
    b.setAttribute('aria-checked', b.dataset.speed === speed ? 'true' : 'false'));
  $('#botSpeedHint').textContent = BOT_SPEEDS[speed].hint;
  scheduleBot();  // a bot already waiting picks up the new speed right away
}

// ------------------------------------------------------------- playback

function setBotsPaused(paused) {
  botsPaused = paused;
  renderPlayback();
  renderStatus();
  scheduleBot();
}

/** Show the position after `turn` turns; null (or the latest turn) returns to the live game. */
function reviewAt(turn) {
  if (!state || !state.replayable) return;
  if (turn !== null) turn = Math.max(0, Math.min(turn, liveTurns));
  reviewTurn = turn === liveTurns ? null : turn;
  payColor = null;
  refresh();
}

function renderPlayback() {
  $('#playback').classList.toggle('hidden', !state);
  if (!state) return;
  const live = !state.review, at = state.turn, can = state.replayable;
  $('#pbBack').disabled = !can || at <= 0;
  $('#pbFwd').disabled = live;
  $('#pbLive').disabled = live;
  $('#pbLabel').textContent = live ? `Move ${liveTurns}` : `Move ${at} / ${liveTurns}`;
  $('#pbLabel').title = can ? '' : 'Replay is only available for games started after this feature was added.';
  const pause = $('#pbPause');
  pause.classList.toggle('paused', botsPaused);
  pause.title = botsPaused ? 'Resume bots (Space)' : 'Pause bots (Space)';
  pause.setAttribute('aria-label', botsPaused ? 'Resume bots' : 'Pause bots');
}

function setupPlayback() {
  $('#pbPause').onclick = () => setBotsPaused(!botsPaused);
  $('#pbBack').onclick = () => reviewAt((state.review ? state.turn : liveTurns) - 1);
  $('#pbFwd').onclick = () => reviewAt(state.turn + 1);
  $('#pbLive').onclick = () => reviewAt(null);
  document.addEventListener('keydown', e => {
    if (!state || e.target.closest('input, select, textarea')) return;
    if (e.key === ' ' && e.target.closest('button')) return;  // Space already presses a focused button
    if (e.key === 'ArrowLeft' && !$('#pbBack').disabled) { e.preventDefault(); $('#pbBack').click(); }
    else if (e.key === 'ArrowRight' && !$('#pbFwd').disabled) { e.preventDefault(); $('#pbFwd').click(); }
    else if (e.key === ' ') { e.preventDefault(); setBotsPaused(!botsPaused); }
  });
}

function setupSettings() {
  const btn = $('#settingsBtn'), menu = $('#settingsMenu');
  const open = show => {
    menu.classList.toggle('hidden', !show);
    btn.setAttribute('aria-expanded', show ? 'true' : 'false');
  };
  btn.onclick = e => { e.stopPropagation(); open(menu.classList.contains('hidden')); };
  document.addEventListener('click', e => { if (!$('#settingsWrap').contains(e.target)) open(false); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') open(false); });
  document.querySelectorAll('#botSpeed button').forEach(b => { b.onclick = () => setBotSpeed(b.dataset.speed); });
  let saved = null;
  try { saved = localStorage.getItem('ttr-bot-speed'); } catch (e) { /* storage blocked */ }
  setBotSpeed(saved || 'normal');
}

async function act(action) {
  if (busy || (state && state.review)) return;
  busy = true;
  try {
    const r = await api.post('/api/action', { player: viewer, action });
    if (r.error) toast(r.error);
  } finally {
    busy = false;
  }
  await refresh();
}

// ---------------------------------------------------------------- render

/**
 * Ways to pay for a route this turn: the fewest locomotives in each color (there's never
 * a reason to spend more), and "all locomotives" only when no color works.
 */
function claimOptions(rid) {
  if (!(myTurn() && state.phase === 'main')) return [];
  const best = new Map();
  for (const [c, n] of state.legal.claims[rid] || []) if (!best.has(c) || n < best.get(c)) best.set(c, n);
  let opts = [...best];
  if (opts.some(([c]) => c !== LOCO)) opts = opts.filter(([c]) => c !== LOCO);
  return opts.sort((a, b) => a[1] - b[1]);
}

function markTicketCities() {
  const mine = new Map();  // city -> 'open' | 'done'
  if (state && state.you && !coverActive()) {
    for (const t of state.you.tickets) {
      for (const c of [t.a, t.b]) if (mine.get(c) !== 'open') mine.set(c, t.done ? 'done' : 'open');
    }
  }
  for (const sel of ['#cities g.city', '#cities text']) {
    d3.selectAll(sel).classed('ticket-open', c => mine.get(c.name) === 'open')
      .classed('ticket-done', c => mine.get(c.name) === 'done');
  }
}

function render() {
  if (!(state && myTurn() && state.phase === 'main' && state.you.hand[payColor] > 0)) payColor = null;
  markTicketCities();
  renderPlayback();
  renderStatus();
  renderPlayers();
  renderMarket();
  renderTray();
  renderRoutes();
  renderLog();
  renderEngine();
  renderModals();
}

// ---------------------------------------------------------- engine view

let hintKey = null, hintData = null, analysisData = null, analyzing = false;

function renderEngine() {
  const available = MAP.bot_levels.includes('expert');
  $('#enginePanel').classList.toggle('hidden', !available || !state);
  if (!available || !state) return;
  const body = $('#engineBody');
  if (!$('#hintToggle').checked) {
    body.innerHTML = '<span class="muted">See how the Expert network rates your options.</span>';
    return;
  }
  if (!myTurn()) { body.innerHTML = '<span class="muted">Shown on your turn.</span>'; return; }
  // Anything that changes after a move (the log alone is capped at 60 entries, so its
  // length stops changing late in the game).
  const key = JSON.stringify([state.phase, state.current, state.deck_size, state.discard_size,
    state.ticket_deck_size, state.face_up, state.route_owner, state.you && state.you.hand,
    state.you && state.you.tickets.length]);
  if (key !== hintKey) {
    hintKey = key; hintData = null; analysisData = null; analyzing = false;
    body.innerHTML = '<span class="muted">Thinking…</span>';
    api.get(`/api/hint?viewer=${viewer}`).then(r => { if (key === hintKey) { hintData = r; drawEngine(); } });
    return;
  }
  drawEngine();
}

function drawEngine() {
  const body = $('#engineBody');
  if (!hintData) return;
  if (hintData.error) { body.innerHTML = `<span class="muted">${esc(hintData.error)}</span>`; return; }
  const v = hintData.value;
  const feel = v > 0.5 ? 'clearly ahead' : v > 0.15 ? 'slightly ahead' : v > -0.15 ? 'about even'
    : v > -0.5 ? 'slightly behind' : 'clearly behind';
  let html = `<div class="eval" title="Roughly: +1 for a win, −1 for a loss, plus the expected score margin ÷ 100">
    Position: <b>${v >= 0 ? '+' : ''}${v.toFixed(2)}</b> (${feel})</div>`;
  const row = (m, i, cols) => `<div class="eng-move ${i === 0 ? 'best' : ''}" data-route="${m.route ?? ''}">${cols}</div>`;
  if (analysisData && analysisData.moves) {
    html += '<div class="eng-head"><span>Move</span><span>Win %</span><span>Margin</span></div>';
    html += analysisData.moves.map((m, i) => row(m, i, `<span>${esc(m.label)}</span>
      <span class="num">${(100 * m.win).toFixed(0)}%</span>
      <span class="num">${m.margin >= 0 ? '+' : ''}${m.margin.toFixed(0)}</span>`)).join('');
    html += `<div class="muted">${analysisData.moves[0].sims} imagined games per move</div>`;
  } else {
    html += '<div class="eng-head"><span>Instinct (first impression)</span><span></span><span>Pref.</span></div>';
    html += hintData.moves.map((m, i) => row(m, i, `<span>${esc(m.label)}
      <div class="bar" style="width:${Math.max(2, 100 * m.prior)}%"></div></span><span></span>
      <span class="num">${(100 * m.prior).toFixed(0)}%</span>`)).join('');
    if (analysisData && analysisData.error) html += `<div class="muted">${esc(analysisData.error)}</div>`;
    html += `<button class="btn" id="analyzeBtn" ${analyzing ? 'disabled' : ''}>
      ${analyzing ? 'Simulating games…' : 'Simulate win % for these moves'}</button>`;
  }
  body.innerHTML = html;
  body.querySelectorAll('.eng-move').forEach(el => {
    const rid = el.dataset.route === '' ? null : +el.dataset.route;
    el.onmouseenter = () => d3.selectAll('#routes g.route').classed('hl', r => r.id === rid);
    el.onmouseleave = () => d3.selectAll('#routes g.route').classed('hl', false);
  });
  const btn = $('#analyzeBtn');
  if (btn) btn.onclick = runAnalysis;
}

async function runAnalysis() {
  const key = hintKey;
  analyzing = true;
  drawEngine();
  const r = await api.post('/api/analyze', { viewer, sims: 48 });
  analyzing = false;
  if (key !== hintKey) return;
  analysisData = r;
  drawEngine();
}

function renderStatus() {
  const el = $('#status');
  if (!state) { el.textContent = 'Start a new game to play.'; return; }
  const cur = state.players[state.current];
  if (state.review) {
    const last = state.log[state.log.length - 1];
    const what = !last || state.turn === 0 ? 'the opening deal'
      : last.player >= 0 ? `${state.players[last.player].name} ${last.text}` : last.text;
    el.innerHTML = `<span class="reviewing">Replay – move ${state.turn} of ${liveTurns}:</span> ${esc(what)}`;
    return;
  }
  let text;
  if (state.phase === 'game_over') text = 'Game over!';
  else if (state.current_is_bot && botsPaused) text = `Bots paused – ${cur.name} is up next.`;
  else if (state.current_is_bot) text = `${cur.name} is thinking…`;
  else if (coverActive()) text = `Pass the device to ${cur.name}.`;
  else if (state.phase === 'setup_tickets') text = `${cur.name}: choose your starting destination tickets.`;
  else if (state.phase === 'choose_tickets') text = `${cur.name}: choose which tickets to keep.`;
  else if (state.phase === 'second_draw') text = `${cur.name}: draw one more train card (no face-up locomotive).`;
  else if (payColor !== null) text = `Paying with ${colorLabel(CARD_TYPES[payColor])}: click a highlighted route (click the card again to cancel).`;
  else text = `${cur.name}: draw cards, claim a route, or draw destination tickets.`;
  el.innerHTML = text + (state.final_turns_left !== null && state.phase !== 'game_over'
    ? `<span class="final">Final round – ${state.final_turns_left} turn${state.final_turns_left === 1 ? '' : 's'} left</span>` : '');
}

function renderPlayers() {
  const box = $('#players');
  box.innerHTML = '';
  if (!state) return;
  for (const p of state.players) {
    const div = document.createElement('div');
    div.className = 'player' + (p.index === state.current && state.phase !== 'game_over' ? ' current' : '');
    div.style.setProperty('--pc', PLAYER_FILL[p.color]);
    const type = p.type === 'human' ? 'Human' : `Bot · ${p.type[0].toUpperCase()}${p.type.slice(1)}`;
    const seen = p.known_hand.map((n, i) => n ? chipHtml(i, n) : '').join('');
    div.innerHTML = `
      <div class="top"><span>${esc(p.name)} <span class="type">${type}</span></span><span>${p.route_score} pts</span></div>
      <div class="stats"><span>🚂 ${p.trains} trains</span><span>🂠 ${p.hand_size} cards</span><span>🎫 ${p.ticket_count} tickets</span></div>
      ${p.index !== viewer && seen ? `<div class="seen">Seen taking: ${seen}</div>` : ''}`;
    box.appendChild(div);
  }
}

function chipHtml(i, n) {
  return `<span class="chip c-${CARD_TYPES[i]}" title="${colorLabel(CARD_TYPES[i])}">${n}</span>`;
}

function cardEl(type, opts = {}) {
  const d = document.createElement('div');
  d.className = `card c-${type} ${type}`;
  d.textContent = type === 'locomotive' ? 'LOCO' : colorLabel(type).toUpperCase();
  if (opts.count !== undefined) {
    const c = document.createElement('span');
    c.className = 'count';
    c.textContent = opts.count;
    d.appendChild(c);
  }
  return d;
}

function renderMarket() {
  const fu = $('#faceUp');
  fu.innerHTML = '';
  if (!state) return;
  const legal = myTurn() ? state.legal : null;
  state.face_up.forEach((c, slot) => {
    if (c === null) { fu.appendChild(document.createElement('div')); return; }
    const el = cardEl(CARD_TYPES[c]);
    const ok = legal && legal.face_up.includes(slot);
    el.classList.add(ok ? 'clickable' : 'disabled');
    if (ok) el.onclick = () => act(['draw_face_up', slot]);
    else if (legal && c === LOCO) el.title = 'You can’t take a face-up locomotive as your second card';
    fu.appendChild(el);
  });
  $('#deckCount').textContent = `${state.deck_size} left · ${state.discard_size} discarded`;
  $('#ticketCount').textContent = `${state.ticket_deck_size} left`;
  $('#deckBtn').disabled = !(legal && legal.draw_deck);
  $('#ticketBtn').disabled = !(legal && legal.draw_tickets && state.phase === 'main');
  $('#passBtn').classList.toggle('hidden', !(legal && legal.pass));
}

function renderTray() {
  const hand = $('#hand'), tix = $('#myTickets');
  hand.innerHTML = '';
  tix.innerHTML = '';
  if (!state || !state.you || coverActive()) {
    $('#handTitle').textContent = 'Your hand';
    hand.innerHTML = '<span class="muted">Hidden</span>';
    return;
  }
  const me = state.players[state.you.index];
  $('#handTitle').textContent = humanCount() > 1 ? `${me.name}’s hand` : 'Your hand';
  const canClaim = myTurn() && state.phase === 'main';
  state.you.hand.forEach((n, i) => {
    if (n <= 0) return;
    const el = cardEl(CARD_TYPES[i], { count: n });
    if (canClaim) {
      el.classList.add('clickable');
      el.title = 'Pay with this card: then click a highlighted route';
      el.onclick = () => { payColor = payColor === i ? null : i; render(); };
    }
    if (payColor === i) el.classList.add('selected');
    hand.appendChild(el);
  });
  if (!hand.children.length) hand.innerHTML = '<span class="muted">No cards</span>';
  for (const t of state.you.tickets) tix.appendChild(ticketEl(t));
  if (!state.you.tickets.length) tix.innerHTML = '<span class="muted">None yet</span>';
}

function ticketEl(t) {
  const d = document.createElement('div');
  d.className = 'ticket' + (t.done ? ' done' : '');
  d.innerHTML = `<div class="cities">${t.a} – ${t.b}</div><span class="pts">${t.points}</span>`;
  d.onmouseenter = () => highlightCities([t.a, t.b]);
  d.onmouseleave = () => highlightCities(selectedTicketCities());
  return d;
}

function highlightCities(names) {
  const set = new Set(names);
  d3.selectAll('#cities g.city').classed('hl', c => set.has(c.name));
  d3.selectAll('#cities text').classed('hl', c => set.has(c.name));
}

function selectedTicketCities() {
  if (modalKind !== 'tickets' || !state.you || !state.you.pending) return [];
  return state.you.pending.filter(t => ticketSelection.has(t.id)).flatMap(t => [t.a, t.b]);
}

function renderRoutes() {
  const owner = state ? state.route_owner : [];
  const recent = new Set(state ? state.last_claims || [] : []);
  d3.selectAll('#routes g.route').each(function (r) {
    const g = d3.select(this);
    g.classed('recent', recent.has(r.id));
    g.classed('claimable', state !== null &&
      claimOptions(r.id).some(([c]) => payColor === null || c === payColor));
    const o = owner[r.id] ?? -1;
    const layer = g.select('g.claimed');
    layer.selectAll('*').remove();
    if (o === -1) return;
    // Plastic train cars: shaded body, raised roof, couplers, and a shadow on the map.
    const color = state.players[o].color;
    layer.attr('filter', 'url(#trainShadow)');
    for (const c of r.geom.cars) {
      const w = c.w + 1, h = CAR_H + 1;  // barely bigger than the spot, so trains at a shared city never touch
      const car = layer.append('g').attr('transform', `translate(${c.x},${c.y - 1.5}) rotate(${c.angle})`);
      car.append('rect').attr('class', 'car-end').attr('x', -w / 2 - 1.5).attr('y', -1.8).attr('width', w + 3).attr('height', 3.6);
      car.append('rect').attr('class', 'car').attr('x', -w / 2).attr('y', -h / 2)
        .attr('width', w).attr('height', h).attr('rx', 3.2).attr('fill', `url(#car-${color})`);
      car.append('rect').attr('class', 'car-roof').attr('x', -w / 2 + 3).attr('y', -h / 2 + 2)
        .attr('width', Math.max(2, w - 6)).attr('height', h * 0.42).attr('rx', 1.8);
    }
  });
}

function renderLog() {
  const ol = $('#log');
  ol.innerHTML = '';
  if (!state) return;
  for (const e of [...state.log].reverse()) {
    const li = document.createElement('li');
    if (e.player >= 0) {
      const p = state.players[e.player];
      li.innerHTML = `<span class="who" style="color:${PLAYER_FILL[p.color] === '#e2b417' ? '#a07c00' : PLAYER_FILL[p.color]}">${esc(p.name)}</span> ${esc(e.text)}`;
    } else {
      li.innerHTML = `<b>${esc(e.text)}</b>`;
    }
    ol.appendChild(li);
  }
}

// ---------------------------------------------------------------- modals

function openModal(kind, html, side = false) {
  modalKind = kind;
  $('#modalBox').innerHTML = html;
  $('#modal').classList.toggle('side', side);
  $('#modal').classList.remove('hidden');
}
function closeModal() {
  modalKind = null;
  $('#modal').classList.add('hidden');
  highlightCities([]);
}

function renderModals() {
  if (!state) return;
  if (state.review) {  // replay is for looking only: put game dialogs away until back to live
    if (['gameover', 'cover', 'tickets', 'claim'].includes(modalKind)) closeModal();
    return;
  }
  if (state.phase === 'game_over') { if (modalKind !== 'gameover' && modalKind !== 'newgame') showGameOver(); return; }
  if (coverActive()) { if (modalKind !== 'cover') showCover(); return; }
  if (modalKind === 'cover') closeModal();
  if (myTurn() && state.you.pending) { if (modalKind !== 'tickets') showTicketChoice(); return; }
  if (modalKind === 'tickets' || modalKind === 'gameover') closeModal();
  if (modalKind === 'claim' && !(myTurn() && state.phase === 'main')) closeModal();
}

function showCover() {
  const p = state.players[state.current];
  openModal('cover', `
    <h2>Pass to ${esc(p.name)}</h2>
    <p>It’s ${esc(p.name)}’s turn. Everyone else, look away!</p>
    <div class="actions"><button class="btn" id="revealBtn">I’m ${esc(p.name)} – show my cards</button></div>`);
  $('#revealBtn').onclick = () => { revealedFor = state.current; closeModal(); render(); };
}

function showTicketChoice() {
  const pending = state.you.pending;
  ticketSelection = new Set(pending.map(t => t.id));
  const min = state.you.min_keep;
  openModal('tickets', `
    <h2>${state.phase === 'setup_tickets' ? 'Starting tickets' : 'New destination tickets'}</h2>
    <p>Keep at least <b>${min}</b>. Click a ticket to toggle it. Kept tickets you don’t complete count against you.</p>
    <div id="pendingTickets" style="display:flex;flex-direction:column;gap:6px"></div>
    <div class="actions"><button class="btn" id="keepBtn">Keep selected</button></div>`, true);
  const box = $('#pendingTickets');
  const update = () => {
    [...box.children].forEach((el, i) => {
      const sel = ticketSelection.has(pending[i].id);
      el.classList.toggle('selected', sel);
      el.classList.toggle('unselected', !sel);
    });
    $('#keepBtn').disabled = ticketSelection.size < min;
    $('#keepBtn').textContent = `Keep ${ticketSelection.size} ticket${ticketSelection.size === 1 ? '' : 's'}`;
    highlightCities(selectedTicketCities());
  };
  for (const t of pending) {
    const el = ticketEl(t);
    el.classList.add('selectable');
    el.onclick = () => {
      ticketSelection.has(t.id) ? ticketSelection.delete(t.id) : ticketSelection.add(t.id);
      update();
    };
    box.appendChild(el);
  }
  $('#keepBtn').onclick = () => { const keep = [...ticketSelection]; closeModal(); act(['keep_tickets', keep]); };
  update();
}

function onRouteClick(r) {
  if (!state || state.phase === 'game_over') return;
  if (state.review) { toast('You’re watching a replay. Press Live to get back to the game.'); return; }
  if (!myTurn()) { toast('It’s not your turn.'); return; }
  if (state.phase !== 'main') { toast(state.phase === 'second_draw' ? 'Draw your second card first.' : 'Choose your tickets first.'); return; }
  const opts = claimOptions(r.id);
  if (!opts.length) { toast(whyNot(r)); return; }
  if (payColor !== null) {
    // Card chosen first: claim straight away with it.
    const opt = opts.find(([c]) => c === payColor);
    if (!opt) { toast(`You can’t pay for ${r.a} – ${r.b} with ${colorLabel(CARD_TYPES[payColor])} cards.`); return; }
    payColor = null;
    act(['claim', r.id, ...opt]);
    return;
  }
  const optHtml = opts.map(([color, locos], i) => {
    const n = r.length - locos;
    const chips = (color === LOCO ? [] : Array(n).fill(color)).concat(Array(color === LOCO ? r.length : locos).fill(LOCO))
      .map(c => `<span class="chip c-${CARD_TYPES[c]}"></span>`).join('');
    const label = color === LOCO ? `${r.length} locomotives`
      : `${n} ${colorLabel(CARD_TYPES[color])}${locos ? ` + ${locos} locomotive${locos > 1 ? 's' : ''}` : ''}`;
    return `<button class="btn secondary pay-option" data-i="${i}">${chips}<span>${label}</span></button>`;
  }).join('');
  openModal('claim', `
    <h2>Claim ${r.a} – ${r.b}</h2>
    <p>${r.length} train${r.length > 1 ? 's' : ''} · scores <b>${ROUTE_POINTS[r.length]}</b> point${r.length > 1 ? 's' : ''}. Pay with:</p>
    ${optHtml}
    <div class="actions"><button class="btn secondary" id="cancelClaim">Cancel</button></div>`);
  d3.selectAll('#modalBox .pay-option').on('click', function () {
    const [color, locos] = opts[+this.dataset.i];
    closeModal();
    act(['claim', r.id, color, locos]);
  });
  $('#cancelClaim').onclick = closeModal;
}

function whyNot(r) {
  const o = state.route_owner[r.id];
  if (o !== -1) return `Already claimed by ${state.players[o].name}.`;
  if (r.pair !== null && state.route_owner[r.pair] !== -1) {
    return state.players.length <= 3 ? 'With 2–3 players only one side of a double route can be used.'
      : 'You can’t claim both sides of a double route.';
  }
  const me = state.players[viewer];
  if (me.trains < r.length) return `You only have ${me.trains} trains left.`;
  return r.color === 'gray'
    ? `You need ${r.length} cards of one color (locomotives are wild).`
    : `You need ${r.length} ${colorLabel(r.color)} cards (locomotives are wild).`;
}

function showGameOver() {
  const res = state.result;
  const globe = state.mode === 'mega';  // Globetrotter bonus for the most completed tickets
  const rows = res.scores.map(s => {
    const p = state.players[s.player];
    return `<tr class="${res.winners.includes(s.player) ? 'winner' : ''}">
      <td><span class="chip" style="background:${PLAYER_FILL[p.color]}"></span> ${esc(p.name)}</td>
      <td>${s.route_points}</td><td>${s.ticket_points >= 0 ? '+' : ''}${s.ticket_points}</td>
      <td>${s.longest_path}${s.longest_bonus ? ' (+10)' : ''}</td>
      ${globe ? `<td>${s.tickets_completed.length}${s.globetrotter_bonus ? ` (+${s.globetrotter_bonus})` : ''}</td>` : ''}
      <td>${s.total}</td></tr>`;
  }).join('');
  const winners = res.winners.map(i => esc(state.players[i].name)).join(' & ');
  const tickets = state.players.map(p => `<div><b>${esc(p.name)}</b><br>${(p.tickets || []).map(t =>
    `<span class="${t.done ? 'ok' : 'bad'}">${t.done ? '✔' : '✘'} ${t.a} – ${t.b} (${t.done ? '+' : '−'}${t.points})</span>`).join('<br>')}</div>`).join('');
  openModal('gameover', `
    <h2>${winners} win${res.winners.length > 1 ? '' : 's'}!</h2>
    <table class="scores"><tr><th>Player</th><th>Routes</th><th>Tickets</th><th>Longest path</th>
      ${globe ? '<th>Completed</th>' : ''}<th>Total</th></tr>${rows}</table>
    <div class="final-tickets">${tickets}</div>
    <div class="actions"><button class="btn secondary" id="viewBoard">View board</button><button class="btn" id="againBtn">New game</button></div>`);
  $('#viewBoard').onclick = () => { closeModal(); modalKind = 'gameover'; };
  $('#againBtn').onclick = showNewGame;
}

const MODE_NAMES = { standard: 'Standard', mega: 'Mega Game (all tickets, Globetrotter bonus)' };

function showNewGame() {
  const defaults = [['You', 'human'], ['Bot 1', 'medium'], ['Bot 2', 'medium'], ['Bot 3', 'medium'], ['Bot 4', 'medium']];
  const levelName = t => (t === 'human' ? 'Human' : `Bot – ${t[0].toUpperCase()}${t.slice(1)}`);
  const seatRow = i => `
    <div class="row seat" data-i="${i}">
      <span class="seat-swatch" style="background:${PLAYER_FILL[MAP.player_colors[i]]}"></span>
      <input class="seat-name" value="${defaults[i][0]}" maxlength="16" size="12">
      <select class="seat-type"></select>
    </div>`;
  openModal('newgame', `
    <h2>New game</h2>
    <div class="row">Board:
      <select id="boardPick">${MAP.boards.map(b => `<option value="${b.key}" ${b.key === MAP.board ? 'selected' : ''}>
        ${esc(b.title)}</option>`).join('')}</select>
    </div>
    <div class="row">Players:
      <select id="numPlayers">${[2, 3, 4, 5].map(n => `<option ${n === 2 ? 'selected' : ''}>${n}</option>`).join('')}</select>
    </div>
    <div class="row" id="modeRow">Game:
      <select id="gameMode"></select>
    </div>
    <div id="seats">${[0, 1, 2, 3, 4].map(seatRow).join('')}</div>
    <label class="row"><input type="checkbox" id="randomOrder"> Random turn order</label>
    <p class="muted">Several humans = pass-and-play on this screen. Bots only see what a player at the table could see.</p>
    <div class="actions">${state ? '<button class="btn secondary" id="cancelNew">Cancel</button>' : ''}<button class="btn" id="startBtn">Start</button></div>`);
  const sync = () => {
    const n = +$('#numPlayers').value;
    document.querySelectorAll('#seats .seat').forEach((el, i) => el.classList.toggle('hidden', i >= n));
  };
  // The board decides which game modes and bot levels are on offer.
  const syncBoard = () => {
    const b = MAP.boards.find(x => x.key === $('#boardPick').value);
    $('#gameMode').innerHTML = b.modes.map(m => `<option value="${m}">${MODE_NAMES[m] || m}</option>`).join('');
    $('#modeRow').classList.toggle('hidden', b.modes.length < 2);
    document.querySelectorAll('#seats .seat-type').forEach((sel, i) => {
      const keep = sel.value || defaults[i][1];
      const options = ['human', ...b.bot_levels];
      sel.innerHTML = options.map(t => `<option value="${t}" ${t === keep ? 'selected' : ''}>${levelName(t)}</option>`).join('');
    });
  };
  $('#numPlayers').onchange = sync;
  $('#boardPick').onchange = syncBoard;
  sync();
  syncBoard();
  if ($('#cancelNew')) $('#cancelNew').onclick = () => { closeModal(); render(); };
  $('#startBtn').onclick = async () => {
    const n = +$('#numPlayers').value;
    const seats = [...document.querySelectorAll('#seats .seat')].slice(0, n).map((el, i) => ({
      name: el.querySelector('.seat-name').value.trim() || `Player ${i + 1}`,
      type: el.querySelector('.seat-type').value,
    }));
    const board = $('#boardPick').value;
    const res = await api.post('/api/new', {
      seats, board, random_order: $('#randomOrder').checked, mode: $('#gameMode').value || 'standard',
    });
    if (res.error) { toast(res.error); return; }
    if (board !== MAP.board) await loadBoard(board);
    viewer = null; lastHuman = null; revealedFor = null; reviewTurn = null;
    closeModal();
    refresh();
  };
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ------------------------------------------------------------------ init

async function init() {
  const first = await api.get('/api/state');
  await loadBoard(first.board);
  $('#deckBtn').onclick = () => act(['draw_deck']);
  $('#ticketBtn').onclick = () => act(['draw_tickets']);
  $('#passBtn').onclick = () => act(['pass']);
  $('#newGameBtn').onclick = showNewGame;
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && payColor !== null) { payColor = null; render(); }
  });
  try { $('#hintToggle').checked = localStorage.getItem('ttr-hints') === '1'; } catch (e) { /* storage blocked */ }
  $('#hintToggle').onchange = () => {
    try { localStorage.setItem('ttr-hints', $('#hintToggle').checked ? '1' : '0'); } catch (e) { /* ignore */ }
    renderEngine();
  };
  setupSettings();
  setupPlayback();
  refresh();
}

init();
