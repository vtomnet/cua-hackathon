import { createVADController, defaultVADOptions, VADOptions } from "../rec/src/vad";

const vad = createVADController();

function json(data: unknown, init: ResponseInit = {}) {
  return new Response(JSON.stringify(data), {
    headers: { "content-type": "application/json" },
    ...init,
  });
}

function parseJSON<T>(req: Request): Promise<T> {
  return req.json() as Promise<T>;
}

Bun.serve({
  port: parseInt(process.env.PORT || "5173"),
  routes: {
    "/api/v1/vad/status": {
      GET: () => json(vad.status()),
    },
    "/api/v1/vad/start": {
      POST: async (req) => {
        const body = (await parseJSON<Partial<VADOptions>>(req).catch(() => ({}))) || {};
        try {
          if (Object.keys(body).length) vad.update(body);
          await vad.start();
          return json({ ok: true, status: vad.status() });
        } catch (e: any) {
          return json({ ok: false, error: String(e?.message || e) }, { status: 400 });
        }
      },
    },
    "/api/v1/vad/stop": {
      POST: async () => {
        try {
          await vad.stop();
          return json({ ok: true, status: vad.status() });
        } catch (e: any) {
          return json({ ok: false, error: String(e?.message || e) }, { status: 400 });
        }
      },
    },
    "/api/v1/vad/options": {
      GET: () => json(defaultVADOptions()),
      PATCH: async (req) => {
        try {
          const body = await parseJSON<Partial<VADOptions>>(req);
          vad.update(body);
          return json({ ok: true });
        } catch (e: any) {
          return json({ ok: false, error: String(e?.message || e) }, { status: 400 });
        }
      },
    },
  },
  development: {
    hmr: true,
    console: true,
  },
});

console.log("VAD server listening on /api/v1/vad/*");

