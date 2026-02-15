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
    if (directMusicBtn) directMusicBtn.disabled = !canGenerate;

    if (recordingState === "idle") {
        recordBtn.innerText = "Start recording";
        recordBtn.disabled = false;
    }

    if (recordingState === "recording") {
        recordBtn.innerText = "Stop recording";
        recordBtn.disabled = false;
    }

    if (recordingState === "recorded") {
        recordBtn.innerText = "Recording complete";
        recordBtn.disabled = true;
    }
}

/* =================================================
   BUTTON HANDLERS
================================================= */

recordBtn?.addEventListener("click", () => {
    if (recordingState === "idle") {
        recordingState = "recording";
        socket.emit("start_recording");
    } else if (recordingState === "recording") {
        recordingState = "recorded";
        socket.emit("stop_recording");
    }
    updateUI();
});

createSongBtn?.addEventListener("click", () => {
    console.log("🎵 Create song (Emotion → Suno) clicked");
    socket.emit("create_song");
});

/* Direct Music RunPod button (EEG → WAV) */
const directMusicBtn = document.getElementById("create-song-direct-btn");
directMusicBtn?.addEventListener("click", () => {
    console.log("🎵 Create song (Direct EEG → Music) clicked");
    socket.emit("create_song_direct");
});

/* Song generation status/result */
socket.on("song_status", (data) => {
    console.log("📊 Song status:", data);
    const statusEl = document.getElementById("song-status");
    if (statusEl) statusEl.textContent = data.message || "";
});

socket.on("song_ready", (data) => {
    console.log("✅ Song ready:", data);
    const statusEl = document.getElementById("song-status");
    if (statusEl) statusEl.textContent = "Ready!";
    const audioEl = document.getElementById("song-audio");
    if (audioEl && data.audio) {
        audioEl.src = data.audio + "?t=" + Date.now();
        audioEl.hidden = false;
        audioEl.play?.();
    }
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
