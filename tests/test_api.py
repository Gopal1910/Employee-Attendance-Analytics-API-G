import asyncio
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pytest
from httpx import AsyncClient, ASGITransport
from pymongo import MongoClient

from app.main import app, db, IST, UTC, init_indexes, round_half_up, compute_work_hours

def trunc_ms(dt: datetime) -> int:
    return (int(dt.timestamp() * 1000) // 1000) * 1000


@pytest.fixture(autouse=True)
def setup_db():
    init_indexes()


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_employee_crud_and_validation():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Invalid regex on emp_code
        res = await client.post("/employees", json={
            "emp_code": "INVALID",
            "name": "Test User",
            "email": "test@example.com",
            "department": "Engineering",
            "joined_on": "2026-01-01"
        })
        assert res.status_code == 422

        # 2. Equal shift_start and shift_end -> 422
        res = await client.post("/employees", json={
            "emp_code": "EMP9901",
            "name": "Shift Test",
            "email": "shift@example.com",
            "department": "Engineering",
            "shift_start": "09:00",
            "shift_end": "09:00",
            "joined_on": "2026-01-01"
        })
        assert res.status_code == 422

        # 3. Create valid employee
        db.employees.delete_one({"emp_code": "EMP9901"})
        res = await client.post("/employees", json={
            "emp_code": "EMP9901",
            "name": "Shift Test",
            "email": "shift@example.com",
            "department": "QA",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-15"
        })
        assert res.status_code == 201
        data = res.json()
        assert data["emp_code"] == "EMP9901"
        assert isinstance(data["created_at"], int)
        assert data["created_at"] > 100000000000
        assert "_id" not in data
        assert "id" not in data

        # 4. Duplicate emp_code -> 409
        res_dup = await client.post("/employees", json={
            "emp_code": "EMP9901",
            "name": "Duplicate Test",
            "email": "dup@example.com",
            "department": "QA",
            "joined_on": "2026-01-15"
        })
        assert res_dup.status_code == 409

        # 5. List employees with pagination and sorting
        res_list = await client.get("/employees?department=QA&page=1&page_size=10")
        assert res_list.status_code == 200
        list_data = res_list.json()
        assert list_data["page"] == 1
        assert list_data["page_size"] == 10
        assert list_data["total"] >= 1
        assert any(e["emp_code"] == "EMP9901" for e in list_data["items"])


@pytest.mark.asyncio
async def test_punch_in_and_rules():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        db.employees.delete_one({"emp_code": "EMP8801"})
        await client.post("/employees", json={
            "emp_code": "EMP8801",
            "name": "Rule Tester",
            "email": "rules@example.com",
            "department": "HR",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": "EMP8801"})

        # Case A: Inside grace period (09:40:00 IST on 2026-08-03, 04:10:00 UTC)
        dt_grace = datetime(2026, 8, 3, 4, 10, 0, tzinfo=UTC)
        res_grace = await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8801",
            "punched_at": trunc_ms(dt_grace),
            "status": "PRESENT"
        })
        assert res_grace.status_code == 201
        data_grace = res_grace.json()
        assert data_grace["date"] == "2026-08-03"
        assert data_grace["late_minutes"] == 0
        assert data_grace["punch_out"] is None
        assert data_grace["work_hours"] is None
        assert data_grace["history"] == []
        assert "_id" not in data_grace and "id" not in data_grace

        # Duplicate on same date -> 409
        res_dup = await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8801",
            "punched_at": trunc_ms(dt_grace),
            "status": "PRESENT"
        })
        assert res_dup.status_code == 409

        # Case B: 1 second beyond grace (09:40:01 IST on 2026-08-04, 04:10:01 UTC) -> late_minutes = 10
        dt_late1 = datetime(2026, 8, 4, 4, 10, 1, tzinfo=UTC)
        res_late1 = await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8801",
            "punched_at": trunc_ms(dt_late1),
            "status": "PRESENT"
        })
        assert res_late1.status_code == 201
        assert res_late1.json()["late_minutes"] == 10

        # Case C: 10:15:59 IST on 2026-08-05 (04:45:59 UTC) -> late_minutes = 45
        dt_late45 = datetime(2026, 8, 5, 4, 45, 59, tzinfo=UTC)
        res_late45 = await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8801",
            "punched_at": trunc_ms(dt_late45),
            "status": "PRESENT"
        })
        assert res_late45.status_code == 201
        assert res_late45.json()["late_minutes"] == 45


@pytest.mark.asyncio
async def test_overnight_shift_punch_in_and_out():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        db.employees.delete_one({"emp_code": "EMP8802"})
        await client.post("/employees", json={
            "emp_code": "EMP8802",
            "name": "Night Owl",
            "email": "night@example.com",
            "department": "Security",
            "shift_start": "22:00",
            "shift_end": "06:00",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": "EMP8802"})

        # Punch in at 01:00 IST on 2026-08-11 (19:30 UTC on 2026-08-10)
        dt_night_in = datetime(2026, 8, 10, 19, 30, 0, tzinfo=UTC)
        res_in = await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8802",
            "punched_at": trunc_ms(dt_night_in),
            "status": "PRESENT"
        })
        assert res_in.status_code == 201
        assert res_in.json()["date"] == "2026-08-10"

        # Punch out at 06:40 IST on 2026-08-11 (01:10 UTC on 2026-08-11)
        dt_night_out = datetime(2026, 8, 11, 1, 10, 0, tzinfo=UTC)
        res_out = await client.post("/attendance/punch-out", json={
            "emp_code": "EMP8802",
            "punched_at": trunc_ms(dt_night_out)
        })
        assert res_out.status_code == 200
        out_data = res_out.json()
        assert out_data["overtime_minutes"] == 40
        assert out_data["work_hours"] == 5.67
        assert out_data["half_day"] is False


@pytest.mark.asyncio
async def test_punch_out_rules_and_half_day():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        db.employees.delete_one({"emp_code": "EMP8803"})
        await client.post("/employees", json={
            "emp_code": "EMP8803",
            "name": "Half Day Tester",
            "email": "half@example.com",
            "department": "Support",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": "EMP8803"})

        # Punch in at 09:30 IST on 2026-08-12 (04:00 UTC)
        dt_in = datetime(2026, 8, 12, 4, 0, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8803",
            "punched_at": trunc_ms(dt_in),
            "status": "PRESENT"
        })

        # Punch out 4 hours later at 13:30 IST (08:00 UTC) -> work_hours = 4.00 (< 4.50 -> half_day = True)
        dt_out = datetime(2026, 8, 12, 8, 0, 0, tzinfo=UTC)
        res_out = await client.post("/attendance/punch-out", json={
            "emp_code": "EMP8803",
            "punched_at": trunc_ms(dt_out)
        })
        assert res_out.status_code == 200
        out_data = res_out.json()
        assert out_data["work_hours"] == 4.00
        assert out_data["half_day"] is True
        assert out_data["overtime_minutes"] == 0


@pytest.mark.asyncio
async def test_regularize_patch():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        dt_new_out = datetime(2026, 8, 12, 13, 0, 0, tzinfo=UTC)
        res_patch = await client.patch("/attendance/EMP8803/2026-08-12", json={
            "punch_out": trunc_ms(dt_new_out),
            "reason": "Corrected missed checkout",
            "regularized_by": "manager.jane"
        })
        assert res_patch.status_code == 200
        patch_data = res_patch.json()
        assert patch_data["work_hours"] == 9.0
        assert patch_data["half_day"] is False
        assert len(patch_data["history"]) == 1
        h0 = patch_data["history"][0]
        assert h0["by"] == "manager.jane"
        assert h0["reason"] == "Corrected missed checkout"
        assert "work_hours" in h0["changes"]
        assert "half_day" in h0["changes"]

        # Setting status to LEAVE clears punch times
        res_leave = await client.patch("/attendance/EMP8803/2026-08-12", json={
            "status": "LEAVE",
            "reason": "Applied for leave retroactively",
            "regularized_by": "hr.lead"
        })
        assert res_leave.status_code == 200
        leave_data = res_leave.json()
        assert leave_data["status"] == "LEAVE"
        assert leave_data["punch_in"] is None
        assert leave_data["punch_out"] is None
        assert leave_data["work_hours"] is None
        assert len(leave_data["history"]) == 2


@pytest.mark.asyncio
async def test_concurrency_punch_in():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        db.employees.delete_one({"emp_code": "EMP8804"})
        await client.post("/employees", json={
            "emp_code": "EMP8804",
            "name": "Race Tester",
            "email": "race@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": "EMP8804"})

        dt_race = datetime(2026, 8, 17, 4, 0, 0, tzinfo=UTC)
        payload = {
            "emp_code": "EMP8804",
            "punched_at": trunc_ms(dt_race),
            "status": "PRESENT"
        }

        tasks = [client.post("/attendance/punch-in", json=payload) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        status_codes = [r.status_code for r in responses]
        assert status_codes.count(201) == 1
        assert status_codes.count(409) == 4


@pytest.mark.asyncio
async def test_concurrency_punch_out():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        db.employees.delete_one({"emp_code": "EMP8805"})
        await client.post("/employees", json={
            "emp_code": "EMP8805",
            "name": "Out Race Tester",
            "email": "outrace@example.com",
            "department": "Engineering",
            "shift_start": "09:30",
            "shift_end": "18:30",
            "joined_on": "2026-01-01"
        })
        db.attendance_logs.delete_many({"emp_code": "EMP8805"})

        dt_in = datetime(2026, 8, 18, 4, 0, 0, tzinfo=UTC)
        await client.post("/attendance/punch-in", json={
            "emp_code": "EMP8805",
            "punched_at": trunc_ms(dt_in),
            "status": "PRESENT"
        })

        dt_out = datetime(2026, 8, 18, 13, 0, 0, tzinfo=UTC)
        payload = {
            "emp_code": "EMP8805",
            "punched_at": trunc_ms(dt_out)
        }

        # Send simultaneous punch-out requests
        tasks = [client.post("/attendance/punch-out", json=payload) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        status_codes = [r.status_code for r in responses]
        assert status_codes.count(200) == 1
        assert status_codes.count(409) == 4


@pytest.mark.asyncio
async def test_analytics_and_explain():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_emp = await client.get("/analytics/employees/EMP0001/monthly?month=2026-07")
        assert res_emp.status_code == 200
        emp_data = res_emp.json()
        assert emp_data["emp_code"] == "EMP0001"
        assert emp_data["working_days"] == 23

        res_dept = await client.get("/analytics/departments/summary?month=2026-07")
        assert res_dept.status_code == 200
        dept_data = res_dept.json()
        assert "items" in dept_data

        res_lead = await client.get("/analytics/leaderboard/late?month=2026-07&limit=5")
        assert res_lead.status_code == 200

        res_trend = await client.get("/analytics/departments/Engineering/trend?from=2026-07-01&to=2026-07-07")
        assert res_trend.status_code == 200
        trend_data = res_trend.json()
        assert len(trend_data["items"]) == 7

        explain_eps = [
            "/admin/explain/attendance_list?emp_code=EMP0001",
            "/admin/explain/employee_monthly?emp_code=EMP0001&month=2026-07",
            "/admin/explain/department_summary?month=2026-07",
            "/admin/explain/late_leaderboard?month=2026-07",
            "/admin/explain/department_trend?department=Engineering&from=2026-07-01&to=2026-07-07",
        ]
        for ep in explain_eps:
            res_exp = await client.get(ep)
            assert res_exp.status_code == 200
            data_exp = res_exp.json()
            assert "explain" in data_exp
            assert "collection" in data_exp


@pytest.mark.asyncio
async def test_leaderboard_ties():
    """Test standard competition ranking ($rank) and tie cutoff behavior."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create test department and employees
        test_dept = "TieDept"
        emp1 = "EMP7701"
        emp2 = "EMP7702"
        emp3 = "EMP7703"

        for code, name in [(emp1, "User 1"), (emp2, "User 2"), (emp3, "User 3")]:
            db.employees.delete_one({"emp_code": code})
            await client.post("/employees", json={
                "emp_code": code,
                "name": name,
                "email": f"{code.lower()}@example.com",
                "department": test_dept,
                "shift_start": "09:30",
                "shift_end": "18:30",
                "joined_on": "2026-01-01"
            })
            db.attendance_logs.delete_many({"emp_code": code})

        # Insert attendance records:
        # EMP7701: 50 late minutes (Rank 1)
        # EMP7702: 30 late minutes (Rank 2)
        # EMP7703: 30 late minutes (Rank 2) - TIED with EMP7702!
        db.attendance_logs.insert_one({
            "emp_code": emp1,
            "date": "2026-09-01",
            "status": "PRESENT",
            "late_minutes": 50,
            "punch_in": datetime(2026, 9, 1, 4, 50, 0, tzinfo=UTC),
            "punch_out": None,
            "work_hours": None,
            "overtime_minutes": 0,
            "half_day": False,
            "history": []
        })
        db.attendance_logs.insert_one({
            "emp_code": emp2,
            "date": "2026-09-01",
            "status": "PRESENT",
            "late_minutes": 30,
            "punch_in": datetime(2026, 9, 1, 4, 30, 0, tzinfo=UTC),
            "punch_out": None,
            "work_hours": None,
            "overtime_minutes": 0,
            "half_day": False,
            "history": []
        })
        db.attendance_logs.insert_one({
            "emp_code": emp3,
            "date": "2026-09-01",
            "status": "PRESENT",
            "late_minutes": 30,
            "punch_in": datetime(2026, 9, 1, 4, 30, 0, tzinfo=UTC),
            "punch_out": None,
            "work_hours": None,
            "overtime_minutes": 0,
            "half_day": False,
            "history": []
        })

        # Query leaderboard with limit=2
        # Because EMP7702 and EMP7703 tie at rank 2, both satisfy rank <= 2!
        # Both must be returned (total 3 items)!
        res = await client.get(f"/analytics/leaderboard/late?month=2026-09&limit=2&department={test_dept}")
        assert res.status_code == 200
        items = res.json()["items"]
        assert len(items) == 3
        assert items[0]["rank"] == 1
        assert items[0]["emp_code"] == emp1
        assert items[1]["rank"] == 2
        assert items[2]["rank"] == 2
        # Deterministically sorted by emp_code ascending on ties:
        assert items[1]["emp_code"] == emp2
        assert items[2]["emp_code"] == emp3


def test_rounding_half_up():
    """Verify Decimal half-up rounding vs Python float banker's rounding."""
    # 2.125 in banker's rounding rounds to 2.12, but half-up rounds to 2.13
    assert round(2.125, 2) == 2.12
    assert round_half_up(2.125, 2) == 2.13

    # Rate 4 decimals: 0.12345 in half-up is 0.1235
    assert round_half_up(0.12345, 4) == 0.1235
