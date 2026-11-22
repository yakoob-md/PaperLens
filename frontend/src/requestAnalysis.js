export async function requestAnalysis(url, options, onStep) {
  const response = await fetch(url, {
    ...options,
    headers: { ...options.headers, Accept: "application/x-ndjson" },
  });

  if (!response.headers.get("content-type")?.includes("application/x-ndjson")) {
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map(error => error.msg).join(" ") : data.detail;
      throw new Error(detail || "Analysis could not be started.");
    }
    return data;
  }

  if (!response.body) throw new Error("The analysis stream was unavailable.");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += value ?? "";
      const lines = buffer.split("\n");
      buffer = done ? "" : lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "step") onStep(event.label);
        if (event.type === "error") throw new Error(event.detail);
        if (event.type === "result") return event.data;
      }
      if (done) throw new Error("The connection closed before analysis finished. Please try again.");
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
