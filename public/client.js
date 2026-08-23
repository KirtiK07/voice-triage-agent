// Scaffold-stage connectivity check only -- real mic capture, audio
// streaming, and barge-in UI are build-stage work.
//
// Reconnect-with-backoff is required, not optional: Vercel Hobby closes
// WebSocket connections at a hard 300s limit (see DECISIONS.md "Deploy
// target"), so a real session must reconnect and resubscribe, not just
// give up.

const statusEl = document.getElementById("status");
let delayMs = 1000;

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/api/ws`);

  ws.addEventListener("open", () => {
    statusEl.textContent = "connected";
    delayMs = 1000;
  });

  ws.addEventListener("close", () => {
    statusEl.textContent = `disconnected, reconnecting in ${delayMs}ms…`;
    setTimeout(connect, delayMs);
    delayMs = Math.min(delayMs * 2, 30000);
  });

  ws.addEventListener("error", () => {
    statusEl.textContent = "connection error";
  });
}

connect();
