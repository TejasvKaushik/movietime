// Separate from index.ts so the test can import it without starting a server.

export const toMinutes = (hhmm: string) =>
  Number(hhmm.slice(0, 2)) * 60 + Number(hhmm.slice(3, 5));

/** Closest row by minutes-of-day, measured around the clock: 23:58 is 4 minutes from 00:02. */
export function nearest<T extends { time_hhmm: string }>(rows: T[], time: string): T {
  const want = toMinutes(time);
  let best = rows[0];
  let bestDist = Infinity;
  for (const row of rows) {
    const raw = Math.abs(toMinutes(row.time_hhmm) - want);
    const dist = Math.min(raw, 1440 - raw);
    if (dist < bestDist) {
      best = row;
      bestDist = dist;
    }
  }
  return best;
}
