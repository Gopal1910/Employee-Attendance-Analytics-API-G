# DECISIONS.md

1. **Indexes.** On `attendance_logs`, we created a unique compound index on `(emp_code, date)` to enforce daily uniqueness and concurrency safety. We added `(date: -1, emp_code: 1)` for paginated listing and trend filtering, `(date: 1, late_minutes: 1)` for the leaderboard, and `(emp_code: 1, punch_in: -1)` for punch-out resolution. On `employees`, unique `(emp_code)` and `(department, joined_on)` support lookups and department aggregation. We rejected an index on `(emp_code, date, status, work_hours)` because index write overhead on high-frequency punch events outweighed negligible read benefits.

2. **Punch-in race.** Both requests calculate the attendance date and call `insert_one`. MongoDB atomically accepts the first write and rejects the second via the unique compound index on `(emp_code, date)`. The winner returns 201; our code catches `DuplicateKeyError` and returns 409, eliminating check-then-insert race conditions.

3. **Ties.** We use MongoDB's `$rank` window operator for standard competition ranking (1, 2, 2, 4). Filtering `rank <= limit` applies after ranking, so all employees tied at or below the cutoff are returned (potentially exceeding `limit` rows), ordered deterministically by `emp_code` ascending.

4. **Headcount.** The aggregation pipeline starts on `employees` (`joined_on <= date_to`) and `$lookup`s `attendance_logs`. Employees with zero logs produce an empty array, but grouping by department counts every matching employee via `$sum: 1`.

5. **One thing you would change for 100x data.** We would pre-aggregate daily and monthly department summaries asynchronously using MongoDB change streams, replacing heavy real-time cross-collection aggregations with indexed point lookups.
