// deno test supabase/functions/get-still/
import { assertEquals } from "jsr:@std/assert@1";
import { nearest, toMinutes } from "./nearest.ts";

const rows = ["01:21", "09:15", "23:58"].map((time_hhmm) => ({ time_hhmm }));

Deno.test("toMinutes", () => {
  assertEquals(toMinutes("00:00"), 0);
  assertEquals(toMinutes("23:59"), 1439);
});

Deno.test("exact match wins", () => {
  assertEquals(nearest(rows, "09:15").time_hhmm, "09:15");
});

Deno.test("picks the closer of two neighbours", () => {
  assertEquals(nearest(rows, "08:00").time_hhmm, "09:15");
  assertEquals(nearest(rows, "05:00").time_hhmm, "01:21");
});

Deno.test("distance wraps around midnight", () => {
  // 00:02 is 4 minutes after 23:58 but 79 minutes before 01:21.
  assertEquals(nearest(rows, "00:02").time_hhmm, "23:58");
});

Deno.test("single row is always the answer", () => {
  assertEquals(nearest([{ time_hhmm: "12:00" }], "03:00").time_hhmm, "12:00");
});
