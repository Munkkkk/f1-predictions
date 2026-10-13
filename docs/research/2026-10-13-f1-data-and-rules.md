# F1 2026: data sources and championship rules (research, 2026-10-13)

Verified on 2026-10-13 against primary sources. Anything not verified is marked **UNVERIFIED**.

Key sources:
- FIA 2026 F1 Regulations, Section A (General Regulatory Provisions), Issue 03, 2026-06-25: <https://api.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf> (abbreviated **FIA-A** below)
- FIA 2026 F1 Regulations, Section B (Sporting), Issue 09, 2026-10-01: <https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf>
- Jolpica API (live responses): <https://api.jolpi.ca/ergast/f1/2026/races/>, `/2026/driverstandings/`, `/2026/constructorstandings/`, `/2026/17/sprint/`
- formula1.com calendar <https://www.formula1.com/en/racing/2026> and standings <https://www.formula1.com/en/results/2026/drivers>

Note: from 2026 the points system and championship rules sit in **Section A** (Art. A2.1 to A2.2), not in the Sporting Regulations as before.

## 1. Points system (2026)

### Grand Prix (FIA-A Art. A2.2.1)

Points go only to the top 10. The scale depends on how much of the Scheduled Race Distance the leader completed. No points are awarded unless the leader completes at least 2 full consecutive laps without a Safety Car or VSC.

| Pos | Col 1: 2 laps to <25% | Col 2: 25% to <50% | Col 3: 50% to <75% | **Col 4: >=75% (full)** |
|---|---|---|---|---|
| 1 | 6 | 13 | 19 | **25** |
| 2 | 4 | 10 | 14 | **18** |
| 3 | 3 | 8 | 12 | **15** |
| 4 | 2 | 6 | 10 | **12** |
| 5 | 1 | 5 | 8 | **10** |
| 6 | - | 4 | 6 | **8** |
| 7 | - | 3 | 4 | **6** |
| 8 | - | 2 | 3 | **4** |
| 9 | - | 1 | 2 | **2** |
| 10 | - | - | 1 | **1** |

The PDF table layout is garbled when extracted as text. The columns above are my reconstruction of the extracted numbers. Columns 1 and 4 are unambiguous. Columns 2 and 3 match the rule in force since 2022.

**Fastest lap: there is no bonus point in 2026.** FIA-A A2.2 gives no fastest-lap point (it was dropped from 2025). Secondary confirmation: <https://www.redbull.com/int-en/f1-sprint-format-everything-you-need-to-know>.

### Sprint (FIA-A Art. A2.2.2)

| Pos | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Pts | 8 | 7 | 6 | 5 | 4 | 3 | 2 | 1 |

- If the leader has completed less than 50% of the Scheduled Sprint Distance, **no points** are awarded. There is no half-points tier.
- Dead heat (FIA-A A2.2.3): the points for the tied positions are added together and shared equally.
- Cross-check: the live Jolpica `2026/17/sprint` response shows 8-7-6-5-4-3-2-1.

## 2. 2026 calendar

The original calendar had 24 rounds. The April **Bahrain** and **Saudi Arabian** GPs were cancelled because of the Middle East conflict. Bahrain was then re-run as the "Gulf Air Bahrain Grand Prix in Malaysia" at Sepang on 2 to 4 October. Saudi Arabia was not replaced, so the season has **23 rounds**. Sources: formula1.com calendar; secondary reports <https://www.gpfans.com/en/f1-news/1088263/f1-bahrain-grand-prix-confirmed-2026-return-malaysia-sepang/> and <https://english.ahram.org.eg/News/573503.aspx>.

Race dates below come from Jolpica `2026/races` (`total`=23). Weekend dates on formula1.com/en/racing/2026 agree. Sprint weekends come from the official announcement <https://www.formula1.com/en/latest/article/formula-1-and-fia-announce-2026-sprint-calendar.3PyLPAazrBNe8kQIS3wOfY>, and Jolpica flags the same six rounds.

| Rd | Grand Prix | Circuit | Race date | Sprint | Status 2026-10-13 |
|---|---|---|---|---|---|
| 1 | Australia | Albert Park | 2026-03-08 | | done |
| 2 | China | Shanghai | 2026-03-15 | **Yes** | done |
| 3 | Japan | Suzuka | 2026-03-29 | | done |
| 4 | Miami | Miami Intl Autodrome | 2026-05-03 | **Yes** | done |
| 5 | Canada | Gilles Villeneuve | 2026-05-24 | **Yes** | done |
| 6 | Monaco | Monaco | 2026-06-07 | | done |
| 7 | Barcelona | Barcelona-Catalunya | 2026-06-14 | | done |
| 8 | Austria | Red Bull Ring | 2026-06-28 | | done |
| 9 | Great Britain | Silverstone | 2026-07-05 | **Yes** | done |
| 10 | Belgium | Spa | 2026-07-19 | | done |
| 11 | Hungary | Hungaroring | 2026-07-26 | | done |
| 12 | Netherlands | Zandvoort | 2026-08-23 | **Yes** | done |
| 13 | Italy | Monza | 2026-09-06 | | done |
| 14 | Spain | Madring (Madrid) | 2026-09-13 | | done |
| 15 | Azerbaijan | Baku | 2026-09-26 (Sat) | | done |
| 16 | Bahrain GP in Malaysia | Sepang | 2026-10-04 | | done |
| 17 | Singapore | Marina Bay | 2026-10-11 | **Yes** | done |
| 18 | United States | COTA | 2026-10-25 | | **remaining** |
| 19 | Mexico City | Hermanos Rodriguez | 2026-11-01 | | **remaining** |
| 20 | Brazil | Interlagos | 2026-11-08 | | **remaining** |
| 21 | Las Vegas | Las Vegas Strip | 2026-11-22 | | **remaining** |
| 22 | Qatar | Lusail | 2026-11-29 | | **remaining** |
| 23 | Abu Dhabi | Yas Marina | 2026-12-06 | | **remaining** |

**6 Grands Prix remain and no sprints remain**, since all six sprints have been run. The most a driver can still score is 6 x 25 = **150 points**.

- **UNVERIFIED:** Reports from July 2026 said the Qatar and Abu Dhabi rounds were "under review" over regional instability. formula1.com still lists both as upcoming, and I found no confirmation of a change. Re-check before running the simulation.
- **UNVERIFIED:** I found no primary FIA/F1 statement saying Saudi Arabia is cancelled outright rather than postponed. That claim comes from secondary press.

## 3. Tie-break for equal points (FIA-A Art. A2.1, "World Championship Titles", item c)

When drivers (or teams) finish on equal points, the higher place goes to:
1. the one with the most **race wins** (first places in a race);
2. if still level, the most second places, then the most thirds, and so on until the tie is broken;
3. if that still fails, the same countback is applied to **qualifying results** during the season.

The text says "first places in a race". I read this as counting Grand Prix finishes only, not Sprint finishes, but that is an interpretation. formula1.com states that a Sprint win is not a GP victory. For Monte Carlo, implement the countback over GP finishing positions, then over qualifying positions.

## 4. Drivers' standings after Round 17 (Singapore, 2026-10-11)

Sources: <https://api.jolpi.ca/ergast/f1/2026/driverstandings/> (round=17) and <https://www.formula1.com/en/results/2026/drivers>. The two agree.

| Pos | Driver | Team | Pts | Wins (Jolpica) |
|---|---|---|---|---|
| 1 | Andrea Kimi Antonelli | Mercedes | 340 | 8 |
| 2 | George Russell | Mercedes | 238 | 3 |
| 3 | Lewis Hamilton | Ferrari | 229 | 1 |
| 4 | Max Verstappen | Red Bull | 221 | 2 |
| 5 | Charles Leclerc | Ferrari | 215 | 1 |
| 6 | Lando Norris | McLaren | 201 | 2 |
| 7 | Oscar Piastri | McLaren | 138 | 0 |
| 8 | Isack Hadjar | Red Bull | 97 | 0 |
| 9 | Liam Lawson | Racing Bulls | 75 | 0 |
| 10 | Pierre Gasly | Alpine | 41 | 0 |

Antonelli leads by 102 points with 150 still available, so the title is not yet mathematically decided. Russell, Hamilton, Verstappen, Leclerc and Norris can all still reach 340 or more on paper. The grid has 11 teams and 22 cars, including Cadillac and Audi (source: `2026/constructorstandings`, 11 entries).

## 5. Jolpica F1 API (Ergast successor)

| Item | Value | Source |
|---|---|---|
| Base URL | `https://api.jolpi.ca/ergast/f1/` (Ergast-compatible paths) | <https://github.com/jolpica/jolpica-f1> |
| Format | JSON with root `MRData` (contains `limit`, `offset`, `total`, `series`, `url`). Paths end with `/` or `.json` | <https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md> |
| Schedule | `/{season}/races/` (Sprint/Qualifying session dates included) | README; live response |
| Race results | `/{season}/results/`, `/{season}/{round}/results/` | README |
| Sprint results | `/{season}/sprint/`, `/{season}/{round}/sprint/` | README; live response |
| Qualifying | `/{season}/qualifying/`, `/{season}/{round}/qualifying/` | README |
| Driver standings | `/{season}/driverstandings/` (optionally `/{season}/{round}/driverstandings/`) | README |
| Constructor standings | `/{season}/constructorstandings/` | README |
| Others | laps, pitstops, circuits, drivers, constructors, seasons, status | README |
| Pagination | `limit` defaults to 30, maximum **100**; `offset` defaults to 0 | README |
| Rate limits (no token) | **4 req/s burst, 500 req/hour sustained**, shared per IP/token; HTTP 429 when exceeded. Limits "will decrease in the future" | <https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md> |
| User-Agent | A custom User-Agent with app name and version is required | README |
| Other | DB dumps downloadable; volunteer-run; Terms of Use in `TERMS.md` | repo |

- **2026 data is available.** On 2026-10-13 it already held complete Round 17 results, sprint and standings (Singapore was raced 2026-10-11).
- **Results bulk queries:** `/{season}/results/` returns one row per driver per race, which is about 22 x 17 = 374 rows. With `limit=100` that takes several pages, so loop on `offset` until you reach `total`.
- **Discrepancy:** FastF1 docs say the server maximum limit is 1000. The Jolpica README says 100. Code against 100.
- **UNVERIFIED: update latency after a race.** No official SLA or figure exists. The data was there within 2 days of Singapore. Assume "hours to ~1 day" and add a staleness check by comparing `round` against the calendar.

## 6. FastF1

Sources: <https://docs.fastf1.dev/>, <https://docs.fastf1.dev/fastf1.html>, <https://docs.fastf1.dev/api_reference/jolpica.html> (docs version 3.8.3).

- **Provides** lap timing, tyre data, weather, car telemetry and position data, session results and the event schedule, all as extended pandas DataFrames. Full timing/telemetry support covers **2018 to now**. Seasons before 2018 fall back to the Ergast/Jolpica backend, which offers results only.
- **Wraps Jolpica:** yes. `fastf1.ergast.Ergast` covers all Jolpica endpoints and keeps the "ergast" name for backward compatibility. The original Ergast service shut down at the start of 2025. Results come as raw JSON, as `ErgastSimpleResponse`, or as `ErgastMultiResponse` (pandas). Pagination is manual, using `is_complete`, `total_results` and `get_next_result_page()`.
- **Caching** is on by default, and the docs strongly discourage disabling it. The directory is set by `fastf1.Cache.enable_cache(dir)` (the directory must exist), then the `FASTF1_CACHE` env var, then the OS default (Windows `%LOCALAPPDATA%\Temp\fastf1`). Cached requests do not count toward rate limits. FastF1 applies soft throttling and raises `RateLimitExceededError` on hard limits.
- **For this project:** plain HTTP against Jolpica is enough for results, standings and the schedule. FastF1 is worth adding only if lap or pace features are needed.

## 7. 2026 regulation reset and what it means for using history

Sources: <https://www.formula1.com/en/latest/article/fia-unveils-formula-1-regulations-for-2026-and-beyond-featuring-more-agile.75qJiYOHXgeJqsVQtDr2UB>, <https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-new-power-units.14jfv7a36905uDJDdNyfQd>, <https://www.fia.com/news/f1s-new-era-everything-you-need-know-about-how-fia-making-formula-1-more-competitive-more>. These were verified via search snippets because the full pages did not load.

- **New power units:** the 1.6 L V6 turbo stays but the **MGU-H is removed**. Electrical power is about 50% of the total, with the MGU-K at 350 kW (up from 120 kW). Cars run 100% advanced sustainable fuel. Boost and Overtake modes are new.
- **New chassis and aero:** the car is smaller (wheelbase 3400 mm, width 1900 mm) and about 30 kg lighter (768 kg per FIA). **Active aero** (Z-mode/X-mode) replaces DRS. Downforce is down about 30% and drag about 55%.
- **Grid and supplier changes:** an 11th team (Cadillac) joined, and Sauber became Audi (confirmed by Jolpica constructor list). New PU manufacturers also entered, though this was not verified individually here.
- **Implications:** the competitive order was reset. Mercedes leads 2026 by a wide margin, while McLaren, the 2025 champion team, is 3rd. Team strength from before 2026 is a weak prior. Driver skill carries over better than team pace. Use the 17 rounds of 2026 as the main signal, and if older seasons are used at all, give them low weight or use them only for driver-level priors and DNF/variance rates. The FIA's ADUO engine-upgrade mechanism allows in-season PU catch-up, so form can drift during the season. Weighting recent races more heavily is reasonable. Source on ADUO: <https://www.formula1.com/en/latest/article/explained-what-is-f1s-aduo-engine-upgrade-system-and-how-will-it-work.2A67N47ntcdQbKhHOd7GdD>.

## Unverified / open points

1. Whether Qatar (R22) and Abu Dhabi (R23) will go ahead. Both are listed as upcoming on formula1.com, but they were reported as "under review" in July.
2. A primary FIA/F1 source for the outright Saudi Arabia cancellation.
3. How long Jolpica takes to update after a race. No official figure exists.
4. Whether the tie-break counts Sprint finishes. The regulation text says "first places in a race", which implies GP only.
5. FIA-A Issue 03 (June 2026) was the latest Section A I found. A later issue may exist.
