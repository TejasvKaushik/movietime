-- Cinema Time — stills lookup table.
-- Paste into the Supabase SQL editor. Lookup table only: no foreign keys, no relations.
-- Max 1440 rows ever (one per minute of the day), enforced by the primary key.

create table stills (
  time_hhmm      char(5)      primary key,  -- "01:21"
  src            text         not null,      -- full R2 public URL
  title          text         not null,      -- film or show title
  director       text,
  year           smallint,
  film_timestamp text,                       -- timecode in the film, "00:14:33"
  imdb           text,                       -- full URL
  letterboxd     text,                       -- full URL
  source         text,                       -- "confirmed" | "visual_only" | "srt_only"
  vlm_confidence real,
  notes          text,
  created_at     timestamptz  default now(),

  -- The uploader is an automated writer; reject malformed keys at the boundary
  -- rather than trusting the pipeline to have parsed HH:MM correctly.
  constraint time_hhmm_format check (time_hhmm ~ '^([01][0-9]|2[0-3]):[0-5][0-9]$'),
  constraint source_valid     check (source is null or source in ('confirmed', 'visual_only', 'srt_only'))
);

-- Tables created via the SQL editor have RLS off, which leaves the table writable
-- through PostgREST with the public anon key. Turn it on, allow reads only.
alter table stills enable row level security;

create policy "public read" on stills
  for select to anon, authenticated
  using (true);

-- The uploader writes with the service role key, which bypasses RLS. No write policy needed.


-- Test rows so the edge function and the website can be verified before the
-- pipeline exists. Delete once real stills land: delete from stills where notes = 'test row';
insert into stills (time_hhmm, src, title, director, year, film_timestamp, imdb, letterboxd, source, notes) values
  ('01:21', 'https://placehold.co/1920x1080/1a1a1a/e8e8e8?text=01:21', 'Back to the Future', 'Robert Zemeckis', 1985, '00:14:33', 'https://www.imdb.com/title/tt0088763/', 'https://letterboxd.com/film/back-to-the-future/', 'confirmed', 'test row'),
  ('09:15', 'https://placehold.co/1920x1080/1a1a1a/e8e8e8?text=09:15', 'Groundhog Day',      'Harold Ramis',    1993, '00:03:12', 'https://www.imdb.com/title/tt0107048/', 'https://letterboxd.com/film/groundhog-day/',      'visual_only', 'test row'),
  ('23:58', 'https://placehold.co/1920x1080/1a1a1a/e8e8e8?text=23:58', 'Metropolis',         'Fritz Lang',      1927, '00:21:40', 'https://www.imdb.com/title/tt0017136/', 'https://letterboxd.com/film/metropolis/',         'srt_only',   'test row');
