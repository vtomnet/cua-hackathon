const BASE = process.env.SERVER_URL || "http://localhost:5173";

async function call(path: string, init?: RequestInit) {
  const res = await fetch(BASE + path, init);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function main(argv: string[]) {
  const cmd = argv[2];
  if (!cmd || ["help", "-h", "--help"].includes(cmd)) {
    console.log(`Usage: bun run cli <start|stop|status> [key=value ...]

Env:
  SERVER_URL=${BASE}
`);
    process.exit(0);
  }

  if (cmd === "status") {
    const out = await call("/api/v1/vad/status");
    console.log(JSON.stringify(out, null, 2));
    return;
  }
  if (cmd === "stop") {
    const out = await call("/api/v1/vad/stop", { method: "POST" });
    console.log(JSON.stringify(out, null, 2));
    return;
  }
  if (cmd === "start") {
    const body: Record<string, any> = {};
    for (let i = 3; i < argv.length; i++) {
      const [k, v] = (argv[i] || "").split("=");
      if (!k) continue;
      const num = Number(v);
      body[k] = Number.isFinite(num) ? num : v;
    }
    const out = await call("/api/v1/vad/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    console.log(JSON.stringify(out, null, 2));
    return;
  }

  console.error(`Unknown command: ${cmd}`);
  process.exit(1);
}

if (import.meta.main) {
  main(Bun.argv).catch((e) => {
    console.error(e);
    process.exit(1);
  });
}

