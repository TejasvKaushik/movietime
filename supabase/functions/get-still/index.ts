// GET /get-still?time=HH:MM  ->  the stills row closest to that time of day.
// At <=1440 rows a full scan is trivial; no index, no cache. See PROJECT.md section 8.

import { nearest } from "./nearest.ts";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
};

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });

  const time = new URL(req.url).searchParams.get("time") ?? "";
  if (!/^([01]\d|2[0-3]):[0-5]\d$/.test(time)) {
    return json({ error: "time must be HH:MM, 24-hour" }, 400);
  }

  try {
    const key = Deno.env.get("SUPABASE_ANON_KEY")!;
    const res = await fetch(
      `${Deno.env.get("SUPABASE_URL")}/rest/v1/stills?select=*`,
      { headers: { apikey: key, Authorization: `Bearer ${key}` } },
    );
    if (!res.ok) throw new Error(`postgrest ${res.status}: ${await res.text()}`);

    const rows = await res.json();
    if (!rows.length) return json({ error: "no stills" }, 404);

    return json(nearest(rows, time));
  } catch (e) {
    return json({ error: e instanceof Error ? e.message : String(e) }, 500);
  }
});
