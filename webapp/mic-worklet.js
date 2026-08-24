// Runs on the dedicated audio rendering thread, not the main thread --
// deliberate choice over the older ScriptProcessorNode, which runs on
// the main thread and has worse/more variable latency. That matters
// here specifically: this project's whole point is measuring accurate
// barge-in cutoff/recovery latency (see voice_agent/pipeline.py), so the
// mic-capture path shouldn't itself be adding jittery, unmeasured delay.
//
// The AudioContext this runs in must already be created at 16000Hz (see
// client.js) -- Whisper/Silero both require 16kHz (see DECISIONS.md
// "Build stage: STT" for why Piper's 22050Hz output is unrelated to this
// requirement: playback doesn't feed back into VAD/STT).
class MicCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true; // keep the processor alive even with no input yet
    }
    const channelData = input[0]; // mono -- getUserMedia requested channelCount: 1
    const pcm16 = new Int16Array(channelData.length);
    for (let i = 0; i < channelData.length; i++) {
      const clamped = Math.max(-1, Math.min(1, channelData[i]));
      pcm16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    // Transfer the underlying buffer (not copy) -- cheap handoff to the
    // main thread, which forwards it straight to the WebSocket.
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    return true;
  }
}

registerProcessor("mic-capture-processor", MicCaptureProcessor);
