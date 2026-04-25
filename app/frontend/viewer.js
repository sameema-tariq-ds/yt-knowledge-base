async function loadViewer(type, render) {
  const params = new URLSearchParams(window.location.search);
  const handle = params.get("handle");
  const content = document.getElementById("content");

  if (!handle) {
    content.innerHTML = emptyState("Missing handle. Run the pipeline first, then open this page from the dashboard.");
    return;
  }

  try {
    const response = await fetch(`/api/pipeline/${encodeURIComponent(handle)}/${type}`);
    const data = await readJson(response);
    if (!response.ok) {
      throw new Error(data.detail || "Unable to load details.");
    }
    render(data);
  } catch (error) {
    const cached = loadCachedDetails(handle, type);
    if (cached) {
      render(cached);
      const summary = document.getElementById("summary");
      summary.textContent = `${summary.textContent} Cached from the last pipeline run.`;
      return;
    }
    content.innerHTML = emptyState(`${error.message}. Run the pipeline again, then reopen this page from the dashboard.`);
  }
}

async function readJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function loadCachedDetails(handle, type) {
  try {
    const raw = sessionStorage.getItem(`pipelineDetails:${handle}`);
    if (!raw) return null;
    const data = JSON.parse(raw);

    if (type === "videos") {
      const videos = data.videos || [];
      return { handle, count: videos.length, videos };
    }

    if (type === "transcripts") {
      const transcripts = data.transcripts || [];
      return { handle, count: transcripts.length, transcripts };
    }

    if (type === "chunks") {
      const chunks = data.chunks || [];
      return {
        handle,
        video_count: chunks.length,
        chunk_count: chunks.reduce((total, item) => total + Number(item.chunks_created || 0), 0),
        videos: chunks.map((item) => ({
          video_id: item.video_id,
          title: item.title,
          url: item.url,
          chunk_count: item.chunks_created || 0,
          chunks: [
            {
              chunk_id: `${item.video_id || "video"}_summary`,
              chunk_index: "summary",
              start_time: 0,
              timestamp_link: item.url,
              token_count: 0,
              preview: `${item.new_chunks_indexed || 0} new chunks indexed out of ${item.chunks_created || 0} generated chunks.`,
            },
          ],
        })),
      };
    }
  } catch {
    return null;
  }
  return null;
}

function emptyState(message) {
  return `<article class="item"><h2>No data</h2><p>${escapeHtml(message)}</p></article>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
