import asyncio
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pytest
from httpx import AsyncClient, ASGITransport

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


# --------------------------------------------------------------------------- #
# 1. ATTENDANCE BUSINESS-RULE BOUNDARIES
# --------------------------------------------------------------------------- #
def test_boundary_work_hours_half_day_precision():
    """R4 & R5: exact second calculations and half-day boundary at 4.50."""
    t0 = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)

    # 16181 seconds / 3600 = 4.494722... -> 4.49 -> half_day = True
    t_half = t0 + timedelta(seconds=16181)
    wh_half = compute_work_hours(t0, t_half)
    assert wh_half == 4.49
    assert (wh_half < 4.50) is True

    # 16199 seconds / 3600 = 4.499722... -> 4.50 -> half_day = False
    t_full = t0 + timedelta(seconds=16199)
    wh_full = compute_work_hours(t0, t_full)
    assert wh_full == 4.50
    assert (wh_full < 4.50) is False

    # 16200 seconds = exactly 4.50 hours
    t_exact = t0 + timedelta(seconds=16200)
    wh_exact = compute_work_hours(t0, t_exact)
    assert wh_exact == 4.50
    assert (wh_exact < 4.50) is False


def test_boundary_late_minutes_exact_seconds():
    """R2: Grace period 10 minutes strictly, minute floor."""
    shift_start = "09:30"
    att_date = "2026-07-06"

    # Exactly 10:00 (600s) -> 0
    dt_grace = datetime(2026, 7, 6, 4, 10, 0, tzinfo=UTC)
    assert compute_late_minutes(dt_grace, shift_start, att_date) == 0

    # 10:01 (601s) -> 10 minutes late
    dt_late = datetime(2026, 7, 6, 4, 10, 1, tzinfo=UTC)
    assert compute_late_minutes(dt_late, shift_start, att_date) == 10

    # 10:59 (659s) -> 10 minutes late
    dt_late59 = datetime(2026, 7, 6, 4, 10, 59, tzinfo=UTC)
    assert compute_late_minutes(dt_late59, shift_start, att_date) == 10

    # 11:00 (660s) -> 11 minutes late
    dt_late11 = datetime(2026, 7, 6, 4, 11, 0, tzinfo=UTC)
    assert compute_late_minutes(dt_late11, shift_start, att_date) == 11


def test_boundary_overtime_exact_seconds():
    """R3: Overtime threshold 30 minutes, minute floor."""
    shift_start = "09:30"
    shift_end = "18:30"
    att_date = "2026-07-06"

    # 29 min 59 sec -> 0
    dt_under = datetime(2026, 7, 6, 13, 29, 59, tzinfo=UTC)
    assert compute_overtime(dt_under, shift_start, shift_end, att_date) == 0

    # Exactly 30 min 00 sec -> 30
    dt_exact = datetime(2026, 7, 6, 13, 30, 0, tzinfo=UTC)
    assert compute_overtime(dt_exact, shift_start, shift_end, att_date) == 30

    # 30 min 59 sec -> 30
    dt_over = datetime(2026, 7, 6, 13, 30, 59, tzinfo=UTC)
    assert compute_overtime(dt_over, shift_start, shift_end, att_date) == 30


# --------------------------------------------------------------------------- #
# 2. CONCURRENT REGULARIZATIONS & AUDIT HISTORY INTEGRITY
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_concurrent_regularization_audit_integrity():
    """
    Verify concurrent regularizations do not silently lose audit history entries.
    Exactly one succeeds per version via history length optimistic concurrency control.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP7001"
        att_date = "2026-07-06"
        db.employees.delete_one({"emp_code": emp_code})
        db.attendance_logs.delete_one({"emp_code": emp_code, "date": att_date})

        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Audit Test",
            "email": "audit@example.com",
            "department": "Security",
            "joined_on": "2026-01-01"
        })

        t_in = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)
        t_out = datetime(2026, 7, 6, 12, 30, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": emp_code,
            "punched_at": trunc_ms(t_in)
        })
        await client.post("/attendance/punch-out", json={
            "emp_code": emp_code,
            "punched_at": trunc_ms(t_out)
        })

        # Launch 5 concurrent regularizations changing status / punch times
        async def do_reg(idx: int):
            new_out = trunc_ms(t_out + timedelta(minutes=idx * 5 + 5))
            return await client.patch(f"/attendance/{emp_code}/{att_date}", json={
                "punch_out": new_out,
                "reason": f"Correction attempt {idx}",
                "regularized_by": f"admin_{idx}"
            })

        tasks = [do_reg(i) for i in range(1, 6)]
        results = await asyncio.gather(*tasks)

        # Count 200 vs 409
        status_codes = [r.status_code for r in results]
        assert 200 in status_codes
        # Any collision returns 409, and no request loses history
        for code in status_codes:
            assert code in (200, 409)

        # Verify final record history length matches the count of successful 200s
        success_count = status_codes.count(200)
        final_rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": att_date})
        assert len(final_rec["history"]) == success_count
        for h in final_rec["history"]:
            assert "at" in h
            assert "by" in h
            assert "reason" in h
            assert "changes" in h
            assert "punch_out" in h["changes"]


# --------------------------------------------------------------------------- #
# 3. EMPLOYEES WITH ZERO ATTENDANCE RECORDS
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_zero_attendance_records():
    """Verify analytics endpoints correctly handle employees with zero attendance records."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP7002"
        db.employees.delete_one({"emp_code": emp_code})
        db.attendance_logs.delete_many({"emp_code": emp_code})

        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Zero Logs User",
            "email": "zero@example.com",
            "department": "Research",
            "joined_on": "2026-07-01"
        })

        # 1. Employee monthly summary: 23 working days in July 2026, 0 present days
        res_m = await client.get(f"/analytics/employees/{emp_code}/monthly?month=2026-07")
        assert res_m.status_code == 200
        data_m = res_m.json()
        assert data_m["working_days"] == 23
        assert data_m["present_days"] == 0.0
        assert data_m["leave_days"] == 0
        assert data_m["late_count"] == 0
        assert data_m["total_late_minutes"] == 0
        assert data_m["total_overtime_minutes"] == 0
        assert data_m["attendance_pct"] == 0.0

        # 2. Department summary: includes employee in headcount, avg_work_hours is null
        res_d = await client.get("/analytics/departments/summary?month=2026-07&department=Research")
        assert res_d.status_code == 200
        data_d = res_d.json()
        assert len(data_d["items"]) == 1
        item = data_d["items"][0]
        assert item["department"] == "Research"
        assert item["headcount"] >= 1
        assert item["avg_work_hours"] is None
        assert item["present_days"] == 0.0


# --------------------------------------------------------------------------- #
# 4. MID-MONTH JOINERS & EMPLOYEES JOINING AFTER REPORTING PERIOD
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mid_month_and_after_period_joiners():
    """Verify R7 & R9 working days and headcount rules for join dates."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_mid = "EMP7003"
        emp_after = "EMP7004"
        db.employees.delete_many({"emp_code": {"$in": [emp_mid, emp_after]}})
        db.attendance_logs.delete_many({"emp_code": {"$in": [emp_mid, emp_after]}})

        # Mid-month joiner: joined on 2026-07-16 (Thursday)
        # Working days between 2026-07-16 and 2026-07-31:
        # July 16, 17 (2 days)
        # July 20, 21, 22, 23, 24 (5 days)
        # July 27, 28, 29, 30, 31 (5 days)
        # Total = 12 working days
        await client.post("/employees", json={
            "emp_code": emp_mid,
            "name": "Mid Month Joiner",
            "email": "mid@example.com",
            "department": "Logistics",
            "joined_on": "2026-07-16"
        })

        # Punch in on July 16 and July 17 (2 full days)
        for d, day_num in [("2026-07-16", 16), ("2026-07-17", 17)]:
            t_in = datetime(2026, 7, day_num, 4, 0, 0, tzinfo=UTC)
            t_out = datetime(2026, 7, day_num, 13, 0, 0, tzinfo=UTC)
            await client.post("/attendance/punch-in", json={
                "emp_code": emp_mid,
                "punched_at": trunc_ms(t_in)
            })
            await client.post("/attendance/punch-out", json={
                "emp_code": emp_mid,
                "punched_at": trunc_ms(t_out)
            })

        res_mid = await client.get(f"/analytics/employees/{emp_mid}/monthly?month=2026-07")
        assert res_mid.status_code == 200
        d_mid = res_mid.json()
        assert d_mid["working_days"] == 12
        assert d_mid["present_days"] == 2.0
        # attendance_pct = 2.0 / 12 * 100 = 16.666... -> 16.67
        assert d_mid["attendance_pct"] == 16.67

        # Employee joining after reporting period: joined 2026-08-01
        await client.post("/employees", json={
            "emp_code": emp_after,
            "name": "After Month Joiner",
            "email": "after@example.com",
            "department": "Logistics",
            "joined_on": "2026-08-01"
        })

        # Query for July 2026 -> working_days = 0, attendance_pct = null
        res_after = await client.get(f"/analytics/employees/{emp_after}/monthly?month=2026-07")
        assert res_after.status_code == 200
        d_after = res_after.json()
        assert d_after["working_days"] == 0
        assert d_after["attendance_pct"] is None

        # Department summary for Logistics in July 2026:
        # Headcount must include EMP7003 (joined <= 2026-07-31) but NOT EMP7004 (joined 2026-08-01)
        res_dept = await client.get("/analytics/departments/summary?month=2026-07&department=Logistics")
        assert res_dept.status_code == 200
        dept_item = res_dept.json()["items"][0]
        assert dept_item["headcount"] == 1


# --------------------------------------------------------------------------- #
# 5. DEPARTMENT SUMMARY RECORD-LEVEL AVERAGE CALCULATION
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_department_summary_record_level_averages():
    """
    Verify avg_work_hours is the average across all eligible attendance records,
    not an average of employee averages.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        dept = "FinanceSpecial"
        emp1 = "EMP7005"
        emp2 = "EMP7006"
        db.employees.delete_many({"emp_code": {"$in": [emp1, emp2]}})
        db.attendance_logs.delete_many({"emp_code": {"$in": [emp1, emp2]}})

        await client.post("/employees", json={
            "emp_code": emp1,
            "name": "Fin One",
            "email": "f1@example.com",
            "department": dept,
            "joined_on": "2026-01-01"
        })
        await client.post("/employees", json={
            "emp_code": emp2,
            "name": "Fin Two",
            "email": "f2@example.com",
            "department": dept,
            "joined_on": "2026-01-01"
        })

        # emp1 has 1 record: 10.00 hours
        # emp2 has 3 records: 4.00, 4.00, 4.00 hours (all half days < 4.50)
        # Record-level avg: (10.00 + 4.00 + 4.00 + 4.00) / 4 = 22.00 / 4 = 5.50 hours
        # (Per-employee avg would be (10.00 + 4.00) / 2 = 7.00 hours -> INCORRECT)
        # Present days: 1.0 (emp1) + 0.5 + 0.5 + 0.5 (emp2) = 2.50 days
        t_base = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)

        # emp1 record
        db.attendance_logs.insert_one({
            "emp_code": emp1,
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": t_base,
            "punch_out": t_base + timedelta(hours=10),
            "work_hours": 10.00,
            "late_minutes": 0,
            "overtime_minutes": 60,
            "half_day": False,
            "history": []
        })

        # emp2 records (3 half-days)
        for i, dt_str in enumerate(["2026-07-07", "2026-07-08", "2026-07-09"]):
            db.attendance_logs.insert_one({
                "emp_code": emp2,
                "date": dt_str,
                "status": "PRESENT",
                "punch_in": t_base,
                "punch_out": t_base + timedelta(hours=4),
                "work_hours": 4.00,
                "late_minutes": 0,
                "overtime_minutes": 0,
                "half_day": True,
                "history": []
            })

        res = await client.get(f"/analytics/departments/summary?month=2026-07&department={dept}")
        assert res.status_code == 200
        item = res.json()["items"][0]
        assert item["headcount"] == 2
        assert item["present_days"] == 2.5
        assert item["avg_work_hours"] == 5.50


# --------------------------------------------------------------------------- #
# 6. LEADERBOARD TIES AT REQUESTED LIMIT
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_leaderboard_ties_at_limit():
    """
    Verify standard competition ranking ($rank) and tie handling:
    When multiple employees tie at rank <= limit, all tied employees are returned.
    Next rank after a tie is skipped (1, 2, 2, 4).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        dept = "TieDept"
        emps = ["EMP7010", "EMP7011", "EMP7012", "EMP7013"]
        db.employees.delete_many({"emp_code": {"$in": emps}})
        db.attendance_logs.delete_many({"emp_code": {"$in": emps}})

        for code in emps:
            await client.post("/employees", json={
                "emp_code": code,
                "name": f"User {code}",
                "email": f"{code.lower()}@example.com",
                "department": dept,
                "joined_on": "2026-01-01"
            })

        # Seed late minutes:
        # EMP7010: 60 min (Rank 1)
        # EMP7011: 45 min (Rank 2)
        # EMP7012: 45 min (Rank 2 - tie)
        # EMP7013: 15 min (Rank 4 - rank 3 skipped)
        t_dummy = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)
        lates = [("EMP7010", 60), ("EMP7011", 45), ("EMP7012", 45), ("EMP7013", 15)]
        for code, l_min in lates:
            db.attendance_logs.insert_one({
                "emp_code": code,
                "date": "2026-07-06",
                "status": "PRESENT",
                "punch_in": t_dummy,
                "punch_out": t_dummy + timedelta(hours=9),
                "work_hours": 9.00,
                "late_minutes": l_min,
                "overtime_minutes": 0,
                "half_day": False,
                "history": []
            })

        # Query with limit=2: both rank 2 employees must be returned -> total 3 rows!
        res_lim2 = await client.get(f"/analytics/leaderboard/late?month=2026-07&limit=2&department={dept}")
        assert res_lim2.status_code == 200
        items2 = res_lim2.json()["items"]
        assert len(items2) == 3
        assert [i["rank"] for i in items2] == [1, 2, 2]
        assert items2[0]["emp_code"] == "EMP7010"
        # Secondary sort by emp_code ascending on ties
        assert items2[1]["emp_code"] == "EMP7011"
        assert items2[2]["emp_code"] == "EMP7012"

        # Query with limit=3: rank 4 is excluded (since rank 3 was skipped, no employee has rank 3)
        res_lim3 = await client.get(f"/analytics/leaderboard/late?month=2026-07&limit=3&department={dept}")
        assert res_lim3.status_code == 200
        items3 = res_lim3.json()["items"]
        assert len(items3) == 3  # Rank 4 is NOT <= 3


# --------------------------------------------------------------------------- #
# 7. DEPARTMENT TREND GAP FILLING & 7-DAY MOVING AVERAGE
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_department_trend_gap_filling_and_moving_avg():
    """
    Verify department trend produces exactly one row per calendar day across weekends,
    computes attendance_rate only on working days with headcount > 0,
    and computes 7-day moving average ignoring nulls.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        dept = "TrendDept"
        emp1 = "EMP7020"
        db.employees.delete_many({"emp_code": emp1})
        db.attendance_logs.delete_many({"emp_code": emp1})

        await client.post("/employees", json={
            "emp_code": emp1,
            "name": "Trend Worker",
            "email": "trend@example.com",
            "department": dept,
            "joined_on": "2026-07-06"  # Monday
        })

        # Seed attendance only on Monday 2026-07-06 (1 full day)
        t_base = datetime(2026, 7, 6, 4, 0, 0, tzinfo=UTC)
        db.attendance_logs.insert_one({
            "emp_code": emp1,
            "date": "2026-07-06",
            "status": "PRESENT",
            "punch_in": t_base,
            "punch_out": t_base + timedelta(hours=9),
            "work_hours": 9.00,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": []
        })

        # Request 7 days: 2026-07-06 (Mon) to 2026-07-12 (Sun)
        res = await client.get(f"/analytics/departments/{dept}/trend?from=2026-07-06&to=2026-07-12")
        assert res.status_code == 200
        data = res.json()
        assert data["department"] == dept
        items = data["items"]
        assert len(items) == 7

        # Day 1: 2026-07-06 (Mon) -> 1 present, rate = 1.0, mov_avg = 1.0
        assert items[0]["date"] == "2026-07-06"
        assert items[0]["is_working_day"] is True
        assert items[0]["headcount"] == 1
        assert items[0]["present_count"] == 1.0
        assert items[0]["attendance_rate"] == 1.0
        assert items[0]["moving_avg_7d"] == 1.0

        # Day 2: 2026-07-07 (Tue) -> 0 present, rate = 0.0, mov_avg = mean(1.0, 0.0) = 0.50
        assert items[1]["date"] == "2026-07-07"
        assert items[1]["is_working_day"] is True
        assert items[1]["present_count"] == 0.0
        assert items[1]["attendance_rate"] == 0.0
        assert items[1]["moving_avg_7d"] == 0.50

        # Weekend: 2026-07-11 (Sat) -> is_working_day = False, rate = null
        assert items[5]["date"] == "2026-07-11"
        assert items[5]["is_working_day"] is False
        assert items[5]["attendance_rate"] is None


# --------------------------------------------------------------------------- #
# 8. PAGINATION, FILTERING & TOTAL COUNTS
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pagination_and_total_counts():
    """Verify pagination and filtering consistency across employees and attendance."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        dept = "PageDept"
        codes = [f"EMP703{i}" for i in range(1, 6)]
        db.employees.delete_many({"emp_code": {"$in": codes}})
        for c in codes:
            await client.post("/employees", json={
                "emp_code": c,
                "name": f"Paging User {c}",
                "email": f"{c.lower()}@example.com",
                "department": dept,
                "joined_on": "2026-01-01"
            })

        # Page 1 of size 2 -> items: 2, total: 5
        r1 = await client.get(f"/employees?department={dept}&page=1&page_size=2")
        assert r1.status_code == 200
        d1 = r1.json()
        assert len(d1["items"]) == 2
        assert d1["total"] == 5
        assert d1["page"] == 1
        assert d1["page_size"] == 2
        assert d1["items"][0]["emp_code"] == "EMP7031"
        assert d1["items"][1]["emp_code"] == "EMP7032"

        # Page 3 of size 2 -> items: 1, total: 5
        r3 = await client.get(f"/employees?department={dept}&page=3&page_size=2")
        assert r3.status_code == 200
        d3 = r3.json()
        assert len(d3["items"]) == 1
        assert d3["total"] == 5
        assert d3["items"][0]["emp_code"] == "EMP7035"

        # Page 4 of size 2 -> items: 0, total: 5
        r4 = await client.get(f"/employees?department={dept}&page=4&page_size=2")
        assert r4.status_code == 200
        d4 = r4.json()
        assert len(d4["items"]) == 0
        assert d4["total"] == 5


# --------------------------------------------------------------------------- #
# 9. EMPTY DATASETS & INVALID REQUESTS
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_empty_datasets_and_invalid_requests():
    """Verify edge case errors: punch-out without punch-in, 24h limit, invalid queries."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        emp_code = "EMP7040"
        db.employees.delete_one({"emp_code": emp_code})
        db.attendance_logs.delete_many({"emp_code": emp_code})

        await client.post("/employees", json={
            "emp_code": emp_code,
            "name": "Edge User",
            "email": "edge@example.com",
            "department": "EdgeDept",
            "joined_on": "2026-01-01"
        })

        # 1. Punch out with no punch-in record -> 404
        r_po_none = await client.post("/attendance/punch-out", json={"emp_code": emp_code})
        assert r_po_none.status_code == 404

        # 2. Punch out when only ABSENT record exists -> 404
        db.attendance_logs.insert_one({
            "emp_code": emp_code,
            "date": "2026-07-06",
            "status": "ABSENT",
            "punch_in": None,
            "punch_out": None,
            "work_hours": None,
            "late_minutes": 0,
            "overtime_minutes": 0,
            "half_day": False,
            "history": []
        })
        r_po_absent = await client.post("/attendance/punch-out", json={"emp_code": emp_code})
        assert r_po_absent.status_code == 404

        # 3. Punch-in followed by punch-out exceeding 24 hours -> 422
        t_in = datetime(2026, 7, 7, 4, 0, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": emp_code,
            "punched_at": trunc_ms(t_in)
        })
        t_out_25h = t_in + timedelta(hours=25)
        r_po_25h = await client.post("/attendance/punch-out", json={
            "emp_code": emp_code,
            "punched_at": trunc_ms(t_out_25h)
        })
        assert r_po_25h.status_code == 422

        # 4. Punch-out with punched_at <= punch_in -> 422
        r_po_before = await client.post("/attendance/punch-out", json={
            "emp_code": emp_code,
            "punched_at": trunc_ms(t_in - timedelta(minutes=1))
        })
        assert r_po_before.status_code == 422

        # 5. Trend endpoint with span > 92 days -> 422
        r_trend_93 = await client.get("/analytics/departments/EdgeDept/trend?from=2026-01-01&to=2026-04-10")
        assert r_trend_93.status_code == 422

        # 6. Trend endpoint with to < from -> 422
        r_trend_rev = await client.get("/analytics/departments/EdgeDept/trend?from=2026-07-10&to=2026-07-01")
        assert r_trend_rev.status_code == 422

        # 7. Trend endpoint with unknown department -> 404
        r_trend_404 = await client.get("/analytics/departments/NonExistentDept/trend?from=2026-07-01&to=2026-07-05")
        assert r_trend_404.status_code == 404

        # 8. Regularize with changes nothing -> 422
        r_patch_same = await client.patch(f"/attendance/{emp_code}/2026-07-07", json={
            "status": "PRESENT",
            "reason": "No actual change",
            "regularized_by": "admin"
        })
        assert r_patch_same.status_code == 422
