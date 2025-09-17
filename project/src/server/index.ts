import { createServer } from "node:http";
import { createVADController, defaultVADOptions, type VADOptions } from "../record/vad";

const vad = createVADController();

const PORT = parseInt(process.env.PORT || "5173");

function sendJSON(res: any, data: unknown, status = 200) {
  res.statusCode = status;
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify(data));
}

async function readJSON<T>(req: any): Promise<T> {
  let raw = "";
  const decoder = new TextDecoder();
  for await (const chunk of req) {
    if (typeof chunk === "string") raw += chunk;
    else raw += decoder.decode(chunk as Uint8Array, { stream: true });
  }
  raw += decoder.decode();
  return raw ? (JSON.parse(raw) as T) : ({} as T);
}

const server = createServer(async (req, res) => {
  const method = (req.method || "GET").toUpperCase();
  const url = new URL(req.url || "/", "http://localhost");
  const path = url.pathname;

  try {
    if (method === "GET" && path === "/api/v1/vad/status") {
      const status = vad.status();
      console.log(`[${new Date().toISOString()}] GET ${path} ->`, status);
      return sendJSON(res, status);
    }

    if (method === "POST" && path === "/api/v1/vad/start") {
      const body = (await readJSON<Partial<VADOptions>>(req).catch(() => ({}))) || {};
      try {
        if (Object.keys(body).length) vad.update(body);
        await vad.start();
        const status = vad.status();
        console.log(`[${new Date().toISOString()}] POST ${path} -> ok`, { body, status });
        return sendJSON(res, { ok: true, status });
      } catch (e: any) {
        console.error(`[${new Date().toISOString()}] POST ${path} error:`, e);
        return sendJSON(res, { ok: false, error: String(e?.message || e) }, 400);
      }
    }

    if (method === "POST" && path === "/api/v1/vad/stop") {
      try {
        await vad.stop();
        const status = vad.status();
        console.log(`[${new Date().toISOString()}] POST ${path} -> ok`, { status });
        return sendJSON(res, { ok: true, status });
      } catch (e: any) {
        console.error(`[${new Date().toISOString()}] POST ${path} error:`, e);
        return sendJSON(res, { ok: false, error: String(e?.message || e) }, 400);
      }
    }

    if (path === "/api/v1/vad/options") {
      if (method === "GET") {
        console.log(`[${new Date().toISOString()}] GET ${path} -> defaults`);
        return sendJSON(res, defaultVADOptions());
      }
      if (method === "PATCH") {
        try {
          const body = await readJSON<Partial<VADOptions>>(req);
          vad.update(body);
          console.log(`[${new Date().toISOString()}] PATCH ${path} -> ok`, body);
          return sendJSON(res, { ok: true });
        } catch (e: any) {
          console.error(`[${new Date().toISOString()}] PATCH ${path} error:`, e);
          return sendJSON(res, { ok: false, error: String(e?.message || e) }, 400);
        }
      }
    }

    res.statusCode = 404;
    res.end("Not Found");
  } catch (err) {
    console.error(`[${new Date().toISOString()}] Unhandled error:`, err);
    res.statusCode = 500;
    res.end("Internal Server Error");
  }
});

server.listen(PORT, () => {
  console.log(`[${new Date().toISOString()}] VAD server listening on http://localhost:${PORT} (PORT=${PORT})`);
});
