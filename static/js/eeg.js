/* =================================================
   SOCKET.IO CONNECTION
================================================= */

const socket = io("/", {
    path: "/socket.io",
    transports: ["websocket"]
});

/* =================================================
   CONSTANTS
================================================= */

const BUFFER_SIZE = 600;
const PLOT_UPDATE_MS = 80;

/* =================================================
   EEG BUFFERS
================================================= */

let x = Array.from({ length: BUFFER_SIZE }, (_, i) => i);
let y1 = Array(BUFFER_SIZE).fill(0);
let y2 = Array(BUFFER_SIZE).fill(0);
let y3 = Array(BUFFER_SIZE).fill(0);

/* =================================================
   PLOT FACTORY
================================================= */

function createPlot(containerId) {
    const trace = {
        x: x,
        y: Array(BUFFER_SIZE).fill(0),
        mode: "lines",
        line: { color: "white", width: 2 }
    };

    const layout = {
        margin: { l: 20, r: 10, t: 10, b: 20 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { visible: false },
        yaxis: { visible: false, range: [-2, 2] }
    };

    Plotly.newPlot(containerId, [trace], layout, {
        displayModeBar: false,
        responsive: true
    });
}

/* =================================================
   CREATE PLOTS
================================================= */

createPlot("plot1");
createPlot("plot2");
createPlot("plot3");

/* =================================================
   SOCKET EVENTS
================================================= */

let lastUpdate = 0;

socket.on("connect", () => {
    console.log("✅ Socket connected");
    socket.emit("request_eeg");
});

socket.on("disconnect", () => {
    console.warn("⚠️ Socket disconnected");
});

/* =================================================
   EEG DATA HANDLER
================================================= */

socket.on("eeg_data", payload => {
    const now = Date.now();
    if (now - lastUpdate < PLOT_UPDATE_MS) return;
    lastUpdate = now;

    if (
        !payload ||
        !Array.isArray(payload.ch1) ||
        !Array.isArray(payload.ch2) ||
        !Array.isArray(payload.ch3)
    ) {
        console.warn("Invalid EEG payload", payload);
        return;
    }

    y1 = y1.slice(payload.ch1.length).concat(payload.ch1);
    y2 = y2.slice(payload.ch2.length).concat(payload.ch2);
    y3 = y3.slice(payload.ch3.length).concat(payload.ch3);

    Plotly.update("plot1", { y: [y1] });
    Plotly.update("plot2", { y: [y2] });
    Plotly.update("plot3", { y: [y3] });
});

/* =================================================
   EMOTION DISPLAY (FROM RUNPOD)
================================================= */

socket.on("emotion_update", data => {
    console.log("🔥 EMOTION EVENT RECEIVED:", data);

    const labelEl = document.getElementById("emotion");
    const dotEl = document.getElementById("emotion-dot");
    const probsListEl = document.getElementById("probs-list");

    if (!labelEl || !dotEl) return;

    const label = (data.label || "UNKNOWN").toUpperCase();
    labelEl.innerText = label;

    const colors = {
        JOY: "#ffd700",
        AMUSEMENT: "#ffcc00",
        INSPIRATION: "#00ffcc",
        TENDERNESS: "#ff99cc",
        NEUTRAL: "#cccccc",
        SADNESS: "#6699ff",
        FEAR: "#9966ff",
        ANGER: "#ff4444",
        DISGUST: "#66aa66",
        ERROR: "#ff0000"
    };

    dotEl.style.background = colors[label] || "white";

    /* Show emotion probabilities */
    if (probsListEl && data.probs && typeof data.probs === "object") {
        const sorted = Object.entries(data.probs)
            .sort((a, b) => b[1] - a[1])
            .map(([name, pct]) => ({
                name: name,
                pct: (pct * 100).toFixed(1),
                bar: Math.round(pct * 100)
            }));
        probsListEl.innerHTML = sorted.map(
            ({ name, pct, bar }) =>
                `<div class="prob-row">
                    <span class="prob-name">${name}</span>
                    <div class="prob-bar-wrap"><div class="prob-bar" style="width:${bar}%"></div></div>
                    <span class="prob-pct">${pct}%</span>
                </div>`
        ).join("");
    }
});

/* =================================================
   RECORDING STATE MACHINE
================================================= */

const recordBtn = document.getElementById("record-btn");
const createSongBtn = document.getElementById("create-song-btn");
const cancelBtn = document.getElementById("cancel-btn");

let recordingState = "idle"; // idle | recording | recorded

function updateUI() {
    if (!recordBtn) return;

    const canGenerate = recordingState === "recorded";
    if (createSongBtn) createSongBtn.disabled = !canGenerate;

    if (recordingState === "idle") {
        recordBtn.innerText = "Start recording";
        recordBtn.disabled = false;
    }

    if (recordingState === "recording") {
        recordBtn.innerText = "Stop recording";
        recordBtn.disabled = false;
    }

    if (recordingState === "recorded") {
        recordBtn.innerText = "Start another recording";
        recordBtn.disabled = false;
    }
}

/* =================================================
   BUTTON HANDLERS
================================================= */

recordBtn?.addEventListener("click", () => {
    if (recordingState === "idle" || recordingState === "recorded") {
        recordingState = "recording";
        socket.emit("start_recording");
    } else if (recordingState === "recording") {
        recordingState = "recorded";
        socket.emit("stop_recording");
    }
    updateUI();
});

createSongBtn?.addEventListener("click", () => {
    const btn = document.getElementById("play-song-btn");
    if (btn) { btn.hidden = true; btn.href = "#"; btn.classList.remove("song-ready"); }
    socket.emit("create_song");
});

/* Song generation status/result */
socket.on("song_status", (data) => {
    const statusEl = document.getElementById("song-status");
    if (statusEl) statusEl.textContent = data.message || "";
});

function showPlayButton(url) {
    const btn = document.getElementById("play-song-btn");
    const statusEl = document.getElementById("song-status");
    const u = (url || "").trim();
    if (!u) return;
    if (btn) {
        btn.href = u;
        btn.hidden = false;
        btn.classList.add("song-ready");
    }
    if (statusEl) statusEl.textContent = "Song ready! Click below to listen.";
}

/* Suno streaming URL – show play button when ready (links to audiopipe.suno.ai) */
socket.on("song_streaming", (data) => {
    const url = data && (data.audio_url || data.url);
    if (url) {
        showPlayButton(url);
        if (data.message) {
            const el = document.getElementById("song-status");
            if (el) el.textContent = data.message;
        }
    }
});

socket.on("song_ready", (data) => {
    const url = data.audio_url || (data.audio ? (window.location.origin + data.audio + "?t=" + Date.now()) : null);
    if (url) showPlayButton(url);
});

socket.on("song_error", (data) => {
    console.error("❌ Song error:", data);
    const statusEl = document.getElementById("song-status");
    if (statusEl) statusEl.textContent = "Error: " + (data.message || "Unknown");
});

cancelBtn?.addEventListener("click", () => {
    recordingState = "idle";
    updateUI();
});

/* =================================================
   INIT
================================================= */

updateUI();
