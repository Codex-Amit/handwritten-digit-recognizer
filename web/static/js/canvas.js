const canvas     = document.getElementById("canvas");
const ctx        = canvas.getContext("2d");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const barChart   = document.getElementById("barChart");
let painting = false;

// Fill with pure white background (not transparent)
ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const src  = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * (canvas.width  / rect.width),
    y: (src.clientY - rect.top)  * (canvas.height / rect.height),
  };
}

function startPaint(e) { e.preventDefault(); painting = true; draw(e); }
function stopPaint()   { painting = false; ctx.beginPath(); }

function draw(e) {
  if (!painting) return;
  e.preventDefault();
  const { x, y } = getPos(e);
  ctx.lineWidth   = 22;
  ctx.lineCap     = "round";
  ctx.lineJoin    = "round";
  ctx.strokeStyle = "#000000";   // pure black on white
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

canvas.addEventListener("mousedown",  startPaint);
canvas.addEventListener("mousemove",  draw);
canvas.addEventListener("mouseup",    stopPaint);
canvas.addEventListener("mouseleave", stopPaint);
canvas.addEventListener("touchstart", startPaint, { passive: false });
canvas.addEventListener("touchmove",  draw,       { passive: false });
canvas.addEventListener("touchend",   stopPaint);

document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  prediction.textContent = "—";
  confidence.textContent = "";
  barChart.innerHTML = "";
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const imageData = canvas.toDataURL("image/png");
  prediction.textContent = "…";
  confidence.textContent = "";

  try {
    const res  = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image: imageData }),
    });
    const data = await res.json();

    if (data.error) {
      prediction.textContent = "!";
      confidence.textContent = data.error;
      return;
    }

    prediction.textContent = data.digit;
    confidence.textContent = `Confidence: ${data.confidence}%`;
    renderBars(data.probabilities, data.digit);
  } catch (err) {
    prediction.textContent = "!";
    confidence.textContent = "Server error";
  }
});

function renderBars(probs, top) {
  barChart.innerHTML = "";
  for (let i = 0; i <= 9; i++) {
    const pct = probs[String(i)] ?? 0;
    const row = document.createElement("div");
    row.className = "bar-row" + (i === top ? " highlight" : "");
    row.innerHTML = `
      <span class="bar-label">${i}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span class="bar-pct">${pct}%</span>`;
    barChart.appendChild(row);
  }
}