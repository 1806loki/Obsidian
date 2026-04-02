## Relative day patterns

- today
- yesterday
- past three days
- this week
- last week
- this month
- last month
- previous month
- past week
- this week so far

## Rolling window patterns (parameterized)

- last `<N>` weeks
- last `<N>` months
- past `<N>` weeks
- past `<N>` months
- past `X` months (same shape as above; different variable name)

## Explicit date patterns

- on `DD/MM/YYYY` OR `MM/DD/YYYY`
- on `<Month Day, Year>` (example shape: “January 11, 2026”)
- from `<any valid date format>` to `<any valid date format>`
- between `<Any valid date format>` to `<Any valid date format>` (same shape as above)

## Time-of-day windows (clock time)

- between `<HH:MM AM/PM>` and `<HH:MM AM/PM>`
- between `<HH:MM>` and `<HH:MM>` (24-hour / no AM/PM variant)
- from `<MM/DD/YYYY HH:MM AM/PM>` (datetime starting point)
- between `<HH:MM AM/PM>` to `<HH:MM AM/PM>` (same intent, different connector word)

## Parts of day (named periods)

- this morning
- this afternoon
- this evening
- tonight

## Comparative time patterns

- compare today vs yesterday
- compare this week vs previous week
- compare this month vs previous month
- compare a specific time window today vs the same time window yesterday

## “Peak / least / busiest” time-period patterns (time-bucketing implied)

These are time-pattern intents that require grouping by time bucket (hour/day/etc.), even when no bucket is specified:

- peak time period
- least busy time period
- highest traffic period
- least traffic period
- most frequent time (within a period like this week)
