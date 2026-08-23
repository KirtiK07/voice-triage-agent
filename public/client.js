// Real mic capture / playback / barge-in client. Protocol matches
// server.py's /api/ws docstring:
//   client -> server: binary frames, raw 16-bit PCM mono @ 16kHz, any
//     chunk size (server buffers into whatever frame size the active
//     VAD needs).
//   server -> client: binary frames = raw 16-bit PCM playback audio at
//     `sample_rate_out` (never assumed -- read from the "ready" event);
//     text frames = JSON events.

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const startButton = document.getElementById("start-button");

let ws = null;
let micContext = null; // AudioContext @ 16kHz, for capture only
let playbackContext = null; // AudioContext @ sample_rate_out, for playback only
let workletNode = null;
let micStream = null;
let sampleRateOut = null;
let nextPlaybackTime = 0;
let scheduledSources = []; // currently playing/queued AudioBufferSourceNodes
let reconnectDelayMs = 1000;

function setStatus(text) {
  statusEl.textContent = text;
}

function logTranscript(text) {
  const line = document.createElement("div");
  line.textContent = text;
  transcriptEl.appendChild(line);
}

// --- Playback: schedules incoming PCM chunks back-to-back for gapless
// audio, but can be cleared instantly on a barge-in -- that instant
// clearing is the actual behavior cutoff_latency (see
// voice_agent/pipeline.py) is designed around, so getting this right
// matters more than it might look for "just play some audio."
function playChunk(pcm16Bytes) {
  if (!playbackContext) return;
  const int16 = new Int16Array(pcm16Bytes);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / (int16[i] < 0 ? 0x8000 : 0x7fff);
  }
  const buffer = playbackContext.createBuffer(1, float32.length, sampleRateOut);
  buffer.copyToChannel(float32, 0);

  const source = playbackContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackContext.destination);

  const now = playbackContext.currentTime;
  const startAt = Math.max(now, nextPlaybackTime);
  source.start(startAt);
  nextPlaybackTime = startAt + buffer.duration;

  scheduledSources.push(source);
  source.onended = () => {
    scheduledSources = scheduledSources.filter((s) => s !== source);
  };
}

function stopPlaybackImmediately() {
  for (const source of scheduledSources) {
    try {
      source.stop();
    } catch {
      // already stopped/ended -- fine
    }
  }
  scheduledSources = [];
  if (playbackContext) {
    nextPlaybackTime = playbackContext.currentTime;
  }
}

// --- WebSocket + protocol handling
function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/api/ws`);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    reconnectDelayMs = 1000;
  });

  ws.addEventListener("message", (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      handleEvent(msg);
    } else {
      playChunk(event.data);
    }
  });

  ws.addEventListener("close", () => {
    setStatus(`disconnected, reconnecting in ${reconnectDelayMs}ms…`);
    // Vercel Hobby closes WebSocket connections at a hard 300s limit
    // (see DECISIONS.md "Deploy target") -- this reconnect logic isn't
    // optional polish, it's required for any call longer than 5 minutes.
    setTimeout(connect, reconnectDelayMs);
    reconnectDelayMs = Math.min(reconnectDelayMs * 2, 30000);
  });

  ws.addEventListener("error", () => {
    setStatus("connection error");
  });
}

function handleEvent(msg) {
  switch (msg.event) {
    case "ready":
      sampleRateOut = msg.sample_rate_out;
      setupPlaybackContext();
      setStatus("listening…");
      break;
    case "barge_in":
      stopPlaybackImmediately();
      setStatus("listening… (interrupted)");
      break;
    case "transcript":
      logTranscript(`You: ${msg.text}`);
      setStatus("thinking…");
      break;
    default:
      break;
  }
}

function setupPlaybackContext() {
  if (playbackContext) return;
  playbackContext = new AudioContext({ sampleRate: sampleRateOut });
  nextPlaybackTime = 0;
}

// --- Microphone capture
async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  // Requesting the AudioContext itself at 16kHz (not just a
  // getUserMedia constraint, which browsers treat as advisory) forces
  // the whole graph -- including this MediaStreamSource -- to run at
  // 16kHz; the browser handles resampling from the hardware's native
  // rate internally. This is what makes the sample-rate contract with
  // the server (see server.py's docstring) actually hold.
  micContext = new AudioContext({ sampleRate: 16000 });
  await micContext.audioWorklet.addModule("mic-worklet.js");

  const source = micContext.createMediaStreamSource(micStream);
  workletNode = new AudioWorkletNode(micContext, "mic-capture-processor");
  workletNode.port.onmessage = (event) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(event.data); // raw Int16 PCM ArrayBuffer
    }
  };
  source.connect(workletNode);
  // Deliberately not connected to micContext.destination -- capture
  // only, no local monitoring/echo of the caller's own voice.
}

startButton.addEventListener("click", async () => {
  startButton.disabled = true;
  setStatus("requesting microphone…");
  try {
    await startMic();
    connect();
  } catch (err) {
    setStatus(`microphone error: ${err.message}`);
    startButton.disabled = false;
  }
});
