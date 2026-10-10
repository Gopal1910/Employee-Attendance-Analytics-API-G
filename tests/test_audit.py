import asyncio
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pytest
from httpx import AsyncClient, ASGITransport
from pymongo import MongoClient

from app.main import (
    app, db, IST, UTC, init_indexes,
    round_half_up, compute_work_hours, compute_overtime, compute_late_minutes,
    calculate_attendance_date, count_working_days
)

def trunc_ms(dt: datetime) -> int:
    return (int(dt.timestamp() * 1000) // 1000) * 1000


@pytest.fixture(autouse=True)
def setup_db():
    init_indexes()


# =========================================================================== #
# STEP 2: BUSINESS RULE BOUNDARIES TESTS
# =========================================================================== #
@pytest.mark.asyncio
async def test_overnight_and_ist_boundaries():
    """Verify R1 overnight shift date assignment across IST boundaries."""
    # Overnight shift: 22:00 - 06:00
    shift_start = "22:00"
    shift_end = "06:00"

    # 1. 21:55:00 IST on 2026-07-06 (16:25:00 UTC) -> belongs to 2026-07-06
    dt1 = datetime(2026, 7, 6, 16, 25, 0, tzinfo=UTC)
    assert calculate_attendance_date(trunc_ms(dt1), shift_start, shift_end) == "2026-07-06"

    # 2. 05:59:59 IST on 2026-07-07 (00:29:59 UTC) -> strictly before 06:00 -> belongs to 2026-07-06
    dt2 = datetime(2026, 7, 7, 0, 29, 59, tzinfo=UTC)
    assert calculate_attendance_date(trunc_ms(dt2), shift_start, shift_end) == "2026-07-06"

    # 3. 06:00:00 IST on 2026-07-07 (00:30:00 UTC) -> at shift end -> belongs to 2026-07-07
    dt3 = datetime(2026, 7, 7, 0, 30, 0, tzinfo=UTC)
    assert calculate_attendance_date(trunc_ms(dt3), shift_start, shift_end) == "2026-07-07"


def test_late_arrival_grace_boundaries():
    """Verify R2 exact 10-minute grace period boundaries."""
    shift_start = "09:30"
    att_date = "2026-07-06"

    # 09:40:00 IST (04:10:00 UTC) -> exactly 10 min elapsed -> 0 late minutes
    dt_grace_exact = datetime(2026, 7, 6, 4, 10, 0, tzinfo=UTC)
    assert compute_late_minutes(dt_grace_exact, shift_start, att_date) == 0

    # 09:40:01 IST (04:10:01 UTC) -> 10 min 1 sec elapsed -> 10 late minutes
    dt_late_1s = datetime(2026, 7, 6, 4, 10, 1, tzinfo=UTC)
    assert compute_late_minutes(dt_late_1s, shift_start, att_date) == 10

    # 09:40:30 IST (04:10:30 UTC) -> 10 min 30 sec elapsed -> 10 late minutes
    dt_late_30s = datetime(2026, 7, 6, 4, 10, 30, tzinfo=UTC)
    assert compute_late_minutes(dt_late_30s, shift_start, att_date) == 10

    # 09:40:59 IST (04:10:59 UTC) -> 10 min 59 sec elapsed -> 10 late minutes
    dt_late_59s = datetime(2026, 7, 6, 4, 10, 59, tzinfo=UTC)
    assert compute_late_minutes(dt_late_59s, shift_start, att_date) == 10

    # 09:41:00 IST (04:11:00 UTC) -> 11 min 0 sec elapsed -> 11 late minutes
    dt_late_11m = datetime(2026, 7, 6, 4, 11, 0, tzinfo=UTC)
    assert compute_late_minutes(dt_late_11m, shift_start, att_date) == 11

    # 10:15:59 IST (04:45:59 UTC) -> 45 min 59 sec elapsed -> 45 late minutes
    dt_late_45m = datetime(2026, 7, 6, 4, 45, 59, tzinfo=UTC)
    assert compute_late_minutes(dt_late_45m, shift_start, att_date) == 45


def test_overtime_boundaries():
    """Verify R3 overtime >= 30 minutes threshold."""
    shift_start = "09:30"
    shift_end = "18:30"
    att_date = "2026-07-06"

    # 18:59:59 IST (13:29:59 UTC) -> 29 min 59 sec -> 0 overtime
    dt_ot_29 = datetime(2026, 7, 6, 13, 29, 59, tzinfo=UTC)
    assert compute_overtime(dt_ot_29, shift_start, shift_end, att_date) == 0

    # 19:00:00 IST (13:30:00 UTC) -> exactly 30 min -> 30 overtime
    dt_ot_30 = datetime(2026, 7, 6, 13, 30, 0, tzinfo=UTC)
    assert compute_overtime(dt_ot_30, shift_start, shift_end, att_date) == 30

    # 19:29:59 IST (13:59:59 UTC) -> 59 min 59 sec -> 59 overtime
    dt_ot_59 = datetime(2026, 7, 6, 13, 59, 59, tzinfo=UTC)
    assert compute_overtime(dt_ot_59, shift_start, shift_end, att_date) == 59


def test_work_hours_half_up_and_half_day_boundary():
    """Verify R4 work-hour half-up rounding and R5 half-day 4.50 boundary."""
    dt_in = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)

    # 1. 2.125 hours (7650s) -> 2.13 (Python round() gives 2.12)
    dt_out_2125 = dt_in + timedelta(seconds=7650)
    wh_2125 = compute_work_hours(dt_in, dt_out_2125)
    assert wh_2125 == 2.13

    # 2. 4.494 hours (16178s) -> 4.49 -> half_day = True (< 4.50)
    dt_out_4494 = dt_in + timedelta(seconds=16178)
    wh_4494 = compute_work_hours(dt_in, dt_out_4494)
    assert wh_4494 == 4.49
    assert wh_4494 < 4.50  # half_day is True

    # 3. 4.495 hours (16182s) -> 4.50 -> half_day = False (NOT < 4.50)
    dt_out_4495 = dt_in + timedelta(seconds=16182)
    wh_4495 = compute_work_hours(dt_in, dt_out_4495)
    assert wh_4495 == 4.50
    assert not (wh_4495 < 4.50)  # half_day is False


@pytest.mark.asyncio
async def test_strict_epoch_millis_types():
    """Verify invalid timestamp types (strings, floats, seconds, negative) return HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # String timestamp
        r1 = await client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": "1783312500000"})
        assert r1.status_code == 422

        # Float timestamp
        r2 = await client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": 1783312500000.5})
        assert r2.status_code == 422

        # Seconds timestamp (below 100000000000)
        r3 = await client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": 1783312500})
        assert r3.status_code == 422

        # Negative timestamp
        r4 = await client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": -100})
        assert r4.status_code == 422


@pytest.mark.asyncio
async def test_date_range_and_pagination_bounds():
    """Verify date_from > date_to and pagination boundary constraints return 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Inverted date range
        r1 = await client.get("/attendance?date_from=2026-07-10&date_to=2026-07-01")
        assert r1.status_code == 422

        # page < 1
        r2 = await client.get("/attendance?page=0")
        assert r2.status_code == 422

        # page_size < 1
        r3 = await client.get("/attendance?page_size=0")
        assert r3.status_code == 422

        # page_size > 100
        r4 = await client.get("/attendance?page_size=101")
        assert r4.status_code == 422


# =========================================================================== #
# STEP 3: CONCURRENCY AUDIT
# =========================================================================== #
@pytest.mark.asyncio
async def test_concurrency_punch_in_10_requests():
    """Verify 10 simultaneous punch-ins produce exactly 1 201 and 9 409s."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP6601"
        db.employees.delete_one({"emp_code": emp_code})
        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Concurrent 1",
            "email": "c1@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": emp_code})

        dt = datetime(2026, 8, 20, 4, 0, 0, tzinfo=UTC)
        payload = {"emp_code": emp_code, "punched_at": trunc_ms(dt), "status": "PRESENT"}

        tasks = [client.post("/attendance/punch-in", json=payload) for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        codes = [r.status_code for r in responses]
        assert codes.count(201) == 1
        assert codes.count(409) == 9

        # Verify DB persisted exactly one record
        assert db.attendance_logs.count_documents({"emp_code": emp_code, "date": "2026-08-20"}) == 1


@pytest.mark.asyncio
async def test_concurrency_punch_out_10_requests():
    """Verify 10 simultaneous punch-outs produce exactly 1 200 and 9 409s."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP6602"
        db.employees.delete_one({"emp_code": emp_code})
        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Concurrent 2",
            "email": "c2@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": emp_code})

        dt_in = datetime(2026, 8, 20, 4, 0, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": emp_code, "punched_at": trunc_ms(dt_in), "status": "PRESENT"
        })

        dt_out = datetime(2026, 8, 20, 13, 0, 0, tzinfo=UTC)
        payload = {"emp_code": emp_code, "punched_at": trunc_ms(dt_out)}

        tasks = [client.post("/attendance/punch-out", json=payload) for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        codes = [r.status_code for r in responses]
        assert codes.count(200) == 1
        assert codes.count(409) == 9

        # Verify record in DB is punched out
        doc = db.attendance_logs.find_one({"emp_code": emp_code, "date": "2026-08-20"})
        assert doc["punch_out"] is not None
        assert doc["work_hours"] == 9.0


@pytest.mark.asyncio
async def test_concurrency_regularization_no_lost_history():
    """Verify concurrent regularizations do not silently overwrite each other's history."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP6603"
        date_str = "2026-08-21"
        db.employees.delete_one({"emp_code": emp_code})
        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Concurrent 3",
            "email": "c3@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": emp_code})

        dt_in = datetime(2026, 8, 21, 4, 0, 0, tzinfo=UTC)
        dt_out = datetime(2026, 8, 21, 13, 0, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": emp_code, "punched_at": trunc_ms(dt_in), "status": "PRESENT"
        })
        await client.post("/attendance/punch-out", json={
            "emp_code": emp_code, "punched_at": trunc_ms(dt_out)
        })

        # Fire 5 concurrent patch requests with different punch_outs
        tasks = []
        for i in range(5):
            t_out = dt_out + timedelta(minutes=i + 1)
            tasks.append(client.patch(f"/attendance/{emp_code}/{date_str}", json={
                "punch_out": trunc_ms(t_out),
                "reason": f"Correction attempt {i}",
                "regularized_by": f"admin_{i}"
            }))

        responses = await asyncio.gather(*tasks)
        codes = [r.status_code for r in responses]
        assert 200 in codes
        # Losers receive 409
        assert all(c in (200, 409) for c in codes)

        # Inspect DB history
        doc = db.attendance_logs.find_one({"emp_code": emp_code, "date": date_str})
        assert len(doc["history"]) == codes.count(200)


# =========================================================================== #
# STEP 4: DELIBERATELY DIFFICULT ANALYTICS DATASET INDEPENDENT VALIDATION
# =========================================================================== #
@pytest.mark.asyncio
async def test_complex_analytics_dataset():
    """
    Construct a difficult dataset with:
    - Mid-month joiner (EMP_D1, joined 2026-07-20)
    - Zero-log employee (EMP_D2, joined 2026-07-01)
    - Late joiner outside reporting month (EMP_D3, joined 2026-08-01)
    - Weekday & weekend logs, half-day logs, leave logs
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        test_dept = "DiffAnalytics"

        # Setup Employees
        for c in ["EMP_D1", "EMP_D2", "EMP_D3"]:
            db.employees.delete_one({"emp_code": c})
            db.attendance_logs.delete_many({"emp_code": c})

        # EMP_D1: Joined 2026-07-20 (Mid-month joiner, 10 working days in July 2026)
        db.employees.insert_one({
            "emp_code": "EMP_D1",
            "name": "Difficult 1",
            "email": "d1@example.com",
            "department": test_dept,
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-07-20",
            "created_at": datetime(2026, 7, 20, tzinfo=UTC)
        })

        # EMP_D2: Joined 2026-07-01 (Zero logs)
        db.employees.insert_one({
            "emp_code": "EMP_D2",
            "name": "Difficult 2 Zero Logs",
            "email": "d2@example.com",
            "department": test_dept,
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-07-01",
            "created_at": datetime(2026, 7, 1, tzinfo=UTC)
        })

        # EMP_D3: Joined 2026-08-01 (Must be EXCLUDED from July reporting period)
        db.employees.insert_one({
            "emp_code": "EMP_D3",
            "name": "Difficult 3 Late Joiner",
            "email": "d3@example.com",
            "department": test_dept,
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-08-01",
            "created_at": datetime(2026, 8, 1, tzinfo=UTC)
        })

        # Logs for EMP_D1:
        # 1. 2026-07-20 (Mon): Full day (9.0h), Present
        db.attendance_logs.insert_one({
            "emp_code": "EMP_D1", "date": "2026-07-20", "status": "PRESENT",
            "punch_in": datetime(2026, 7, 20, 4, 0, tzinfo=UTC),
            "punch_out": datetime(2026, 7, 20, 13, 0, tzinfo=UTC),
            "work_hours": 9.0, "late_minutes": 0, "overtime_minutes": 0, "half_day": False, "history": []
        })
        # 2. 2026-07-21 (Tue): Full day (9.0h), Present
        db.attendance_logs.insert_one({
            "emp_code": "EMP_D1", "date": "2026-07-21", "status": "PRESENT",
            "punch_in": datetime(2026, 7, 21, 4, 0, tzinfo=UTC),
            "punch_out": datetime(2026, 7, 21, 13, 0, tzinfo=UTC),
            "work_hours": 9.0, "late_minutes": 0, "overtime_minutes": 0, "half_day": False, "history": []
        })
        # 3. 2026-07-22 (Wed): Half day (4.0h), WFH (Counts as 0.5 present day)
        db.attendance_logs.insert_one({
            "emp_code": "EMP_D1", "date": "2026-07-22", "status": "WFH",
            "punch_in": datetime(2026, 7, 22, 4, 0, tzinfo=UTC),
            "punch_out": datetime(2026, 7, 22, 8, 0, tzinfo=UTC),
            "work_hours": 4.0, "late_minutes": 0, "overtime_minutes": 0, "half_day": True, "history": []
        })
        # 4. 2026-07-23 (Thu): LEAVE (Does not count as present)
        db.attendance_logs.insert_one({
            "emp_code": "EMP_D1", "date": "2026-07-23", "status": "LEAVE",
            "punch_in": None, "punch_out": None,
            "work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False, "history": []
        })
        # 5. 2026-07-25 (Sat, Weekend): Present, late 20 min, overtime 40 min
        # Weekend does NOT add to present_days, but adds to late & overtime totals!
        db.attendance_logs.insert_one({
            "emp_code": "EMP_D1", "date": "2026-07-25", "status": "PRESENT",
            "punch_in": datetime(2026, 7, 25, 4, 20, tzinfo=UTC),
            "punch_out": datetime(2026, 7, 25, 13, 40, tzinfo=UTC),
            "work_hours": 9.33, "late_minutes": 20, "overtime_minutes": 40, "half_day": False, "history": []
        })

        # --- A. EMP_D1 Monthly Summary ---
        # Expected:
        # working_days: 10
        # present_days: 1.0 (Mon) + 1.0 (Tue) + 0.5 (Wed) = 2.5
        # leave_days: 1
        # late_count: 1 (from Saturday)
        # total_late_minutes: 20
        # total_overtime_minutes: 40
        # attendance_pct: (2.5 / 10) * 100 = 25.0%
        r_m1 = await client.get("/analytics/employees/EMP_D1/monthly?month=2026-07")
        assert r_m1.status_code == 200
        d_m1 = r_m1.json()
        assert d_m1["working_days"] == 10
        assert d_m1["present_days"] == 2.5
        assert d_m1["leave_days"] == 1
        assert d_m1["late_count"] == 1
        assert d_m1["total_late_minutes"] == 20
        assert d_m1["total_overtime_minutes"] == 40
        assert d_m1["attendance_pct"] == 25.0

        # --- B. EMP_D2 Zero Logs Monthly Summary ---
        r_m2 = await client.get("/analytics/employees/EMP_D2/monthly?month=2026-07")
        assert r_m2.status_code == 200
        d_m2 = r_m2.json()
        assert d_m2["working_days"] == 23
        assert d_m2["present_days"] == 0.0
        assert d_m2["leave_days"] == 0
        assert d_m2["late_count"] == 0
        assert d_m2["attendance_pct"] == 0.0

        # --- C. Department Summary ---
        # Headcount: EMP_D1 and EMP_D2 count (EMP_D3 excluded!) -> headcount = 2!
        # present_days: 2.5
        # avg_work_hours: mean of 9.0, 9.0, 4.0, 9.33 (4 records) = 31.33 / 4 = 7.8325 -> 7.83!
        # late_count: 1, total_late_minutes: 20, leave_count: 1
        r_dept = await client.get(f"/analytics/departments/summary?month=2026-07&department={test_dept}")
        assert r_dept.status_code == 200
        dept_items = r_dept.json()["items"]
        assert len(dept_items) == 1
        dept_res = dept_items[0]
        assert dept_res["department"] == test_dept
        assert dept_res["headcount"] == 2  # EMP_D2 with zero logs counted! EMP_D3 excluded!
        assert dept_res["present_days"] == 2.5
        assert dept_res["avg_work_hours"] == 7.83
        assert dept_res["late_count"] == 1
        assert dept_res["total_late_minutes"] == 20
        assert dept_res["leave_count"] == 1
