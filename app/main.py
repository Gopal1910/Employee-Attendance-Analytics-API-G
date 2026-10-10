"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (system environment variables take precedence over local .env)
"""
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator
from pymongo import MongoClient, ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

# Ensure environment variables take precedence over local .env
load_dotenv(override=False)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(MONGO_URI, tz_aware=True, serverSelectionTimeoutMS=2000)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Lifespan & Idempotent Index Initialization
# --------------------------------------------------------------------------- #
def init_indexes():
    """Create all necessary indexes at startup idempotently."""
    # employees collection indexes
    db.employees.create_index([("emp_code", ASCENDING)], unique=True)
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)])
    db.employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)])
    db.employees.create_index([("joined_on", ASCENDING)])

    # attendance_logs collection indexes
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True)
    db.attendance_logs.create_index([("date", DESCENDING), ("emp_code", ASCENDING)])
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("date", DESCENDING)])
    db.attendance_logs.create_index([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)])
    db.attendance_logs.create_index([("date", ASCENDING), ("late_minutes", ASCENDING)])
    db.attendance_logs.create_index([("emp_code", ASCENDING), ("punch_in", DESCENDING)])


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_indexes()
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Helpers & Business Rule Implementations (R1 - R8)
# --------------------------------------------------------------------------- #
def round_half_up(value: float, places: int = 2) -> float:
    """R8: Decimal half-up rounding (avoid Python round-half-to-even)."""
    d = Decimal(str(value))
    target = Decimal("1." + "0" * places if places > 0 else "1")
    target_str = "0." + "0" * places if places > 0 else "1"
    rounded = d.quantize(Decimal(target_str), rounding=ROUND_HALF_UP)
    return float(rounded)


def truncate_instant_ms(ms: int) -> int:
    """R1: Truncate instants to whole seconds before any calculation."""
    return (ms // 1000) * 1000


def ms_to_utc_dt(ms: int) -> datetime:
    """Convert epoch milliseconds to timezone-aware UTC datetime."""
    truncated_ms = truncate_instant_ms(ms)
    return datetime.fromtimestamp(truncated_ms / 1000.0, tz=UTC)


def dt_to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert datetime to epoch milliseconds integer."""
    if dt is None:
        return None
    if isinstance(dt, (int, float)):
        return int(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def parse_hh_mm(s: str) -> tuple[int, int]:
    h, m = map(int, s.split(":"))
    return h, m


def calculate_attendance_date(punch_in_ms: int, shift_start: str, shift_end: str) -> str:
    """
    R1: Attendance date calculation in IST (UTC+05:30).
    For overnight shifts (shift_end <= shift_start), a punch-in earlier than shift_end
    belongs to the shift that started on the previous calendar day.
    """
    punch_dt_utc = ms_to_utc_dt(punch_in_ms)
    punch_dt_ist = punch_dt_utc.astimezone(IST)

    start_h, start_m = parse_hh_mm(shift_start)
    end_h, end_m = parse_hh_mm(shift_end)
    is_overnight = (end_h, end_m) <= (start_h, start_m)

    cal_date = punch_dt_ist.date()
    if is_overnight:
        punch_time = punch_dt_ist.time()
        end_time = time(end_h, end_m)
        if punch_time < end_time:
            cal_date = cal_date - timedelta(days=1)

    return cal_date.isoformat()


def compute_late_minutes(punch_in_dt: datetime, shift_start: str, attendance_date_str: str) -> int:
    """
    R2: Late arrival calculation.
    Grace period is 10 minutes. A punch-in is late only if strictly > 10:00 after shift_start.
    When late, late_minutes = floor elapsed minutes since shift_start (not grace end).
    """
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=UTC)
    punch_in_ist = punch_in_dt.astimezone(IST)

    att_date = date.fromisoformat(attendance_date_str)
    start_h, start_m = parse_hh_mm(shift_start)
    shift_start_dt = datetime.combine(att_date, time(start_h, start_m), tzinfo=IST)

    diff_seconds = (punch_in_ist - shift_start_dt).total_seconds()
    if diff_seconds > 600:  # strictly more than 10 minutes (600 seconds)
        return int(diff_seconds // 60)
    return 0


def compute_work_hours(punch_in_dt: datetime, punch_out_dt: datetime) -> Optional[float]:
    """
    R4: (punch_out - punch_in) in seconds / 3600, rounded to 2 decimals half-up.
    """
    if punch_in_dt is None or punch_out_dt is None:
        return None
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=UTC)
    if punch_out_dt.tzinfo is None:
        punch_out_dt = punch_out_dt.replace(tzinfo=UTC)

    diff_seconds = int((punch_out_dt - punch_in_dt).total_seconds())
    hours = Decimal(diff_seconds) / Decimal(3600)
    return float(hours.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(
    punch_out_dt: datetime, shift_start: str, shift_end: str, attendance_date_str: str
) -> int:
    """
    R3: Overtime calculation.
    Whole minutes (floor) from shift_end to punch_out, counted only when >= 30; otherwise 0.
    Handles overnight shifts where shift_end falls on the next calendar day.
    """
    if punch_out_dt is None:
        return 0
    if punch_out_dt.tzinfo is None:
        punch_out_dt = punch_out_dt.replace(tzinfo=UTC)
    punch_out_ist = punch_out_dt.astimezone(IST)

    att_date = date.fromisoformat(attendance_date_str)
    start_h, start_m = parse_hh_mm(shift_start)
    end_h, end_m = parse_hh_mm(shift_end)

    shift_end_date = att_date
    if (end_h, end_m) <= (start_h, start_m):
        shift_end_date = att_date + timedelta(days=1)

    shift_end_dt = datetime.combine(shift_end_date, time(end_h, end_m), tzinfo=IST)
    diff_seconds = (punch_out_ist - shift_end_dt).total_seconds()
    if diff_seconds >= 1800:  # at least 30 minutes (1800 seconds)
        return int(diff_seconds // 60)
    return 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    """
    R5: After punch-out, half_day = true when rounded work_hours < 4.50.
    """
    if work_hours is None:
        return False
    return work_hours < 4.50


# --------------------------------------------------------------------------- #
# Pydantic Request & Response Models (Authoritative from openapi.yaml)
# --------------------------------------------------------------------------- #
class EmployeeCreate(BaseModel):
    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field("18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def validate_shifts(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class Employee(BaseModel):
    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str
    shift_end: str
    joined_on: str
    created_at: int


class EmployeePage(BaseModel):
    items: List[Employee]
    total: int
    page: int
    page_size: int


class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = Field(None, ge=100000000000, le=4102444800000)
    status: str = Field("PRESENT", pattern=r"^(PRESENT|WFH|ON_DUTY)$")


class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[StrictInt] = Field(None, ge=100000000000, le=4102444800000)


class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$")
    punch_in: Optional[StrictInt] = Field(None, ge=100000000000, le=4102444800000)
    punch_out: Optional[StrictInt] = Field(None, ge=100000000000, le=4102444800000)
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)


class HistoryEntry(BaseModel):
    at: int
    by: str
    reason: str
    changes: Dict[str, Dict[str, Any]]


class AttendanceRecord(BaseModel):
    emp_code: str
    date: str
    status: str
    punch_in: Optional[int]
    punch_out: Optional[int]
    work_hours: Optional[float]
    late_minutes: int
    overtime_minutes: int
    half_day: bool
    history: List[HistoryEntry]


class AttendancePage(BaseModel):
    items: List[AttendanceRecord]
    total: int
    page: int
    page_size: int


class EmployeeMonthly(BaseModel):
    emp_code: str
    month: str
    working_days: int
    present_days: float
    leave_days: int
    late_count: int
    total_late_minutes: int
    total_overtime_minutes: int
    attendance_pct: Optional[float]


class DepartmentSummaryItem(BaseModel):
    department: str
    headcount: int
    present_days: float
    avg_work_hours: Optional[float]
    late_count: int
    total_late_minutes: int
    leave_count: int
    on_duty_count: int


class DepartmentSummary(BaseModel):
    month: str
    items: List[DepartmentSummaryItem]


class LeaderboardItem(BaseModel):
    rank: int
    emp_code: str
    name: str
    department: str
    total_late_minutes: int
    late_count: int


class Leaderboard(BaseModel):
    month: str
    items: List[LeaderboardItem]


class TrendItem(BaseModel):
    date: str
    is_working_day: bool
    headcount: int
    present_count: float
    late_count: int
    attendance_rate: Optional[float]
    moving_avg_7d: Optional[float]


class Trend(BaseModel):
    department: str
    items: List[TrendItem]


# --------------------------------------------------------------------------- #
# Document Formatting Functions
# --------------------------------------------------------------------------- #
def format_employee_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    created_at = doc.get("created_at")
    if isinstance(created_at, datetime):
        created_at_ms = dt_to_epoch_ms(created_at)
    elif isinstance(created_at, (int, float)):
        created_at_ms = int(created_at)
    else:
        created_at_ms = 0

    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": created_at_ms,
    }


def format_attendance_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    history_list = []
    for h in doc.get("history", []) or []:
        changes_api = {}
        for k, v in h.get("changes", {}).items():
            from_val = v.get("from")
            to_val = v.get("to")
            if k in ("punch_in", "punch_out"):
                if isinstance(from_val, datetime):
                    from_val = dt_to_epoch_ms(from_val)
                if isinstance(to_val, datetime):
                    to_val = dt_to_epoch_ms(to_val)
            changes_api[k] = {"from": from_val, "to": to_val}

        at_val = h.get("at")
        at_ms = dt_to_epoch_ms(at_val) if isinstance(at_val, datetime) else int(at_val or 0)

        history_list.append({
            "at": at_ms,
            "by": h.get("by", ""),
            "reason": h.get("reason", ""),
            "changes": changes_api,
        })

    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_epoch_ms(doc.get("punch_in")),
        "punch_out": dt_to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes", 0) or 0,
        "overtime_minutes": doc.get("overtime_minutes", 0) or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": history_list,
    }


# --------------------------------------------------------------------------- #
# Endpoints: System & Employees
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
def health():
    """Readiness probe: Returns 200 only if MongoDB responds, otherwise 503."""
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


@app.post("/employees", response_model=Employee, status_code=status.HTTP_201_CREATED, tags=["employees"])
def create_employee(body: EmployeeCreate):
    """Create an employee. Enforces uniqueness on emp_code via unique index."""
    doc = body.model_dump()
    doc["created_at"] = datetime.now(UTC).replace(microsecond=0)

    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="emp_code already exists")

    return format_employee_doc(doc)


@app.get("/employees", response_model=EmployeePage, tags=["employees"])
def list_employees(
    department: Optional[str] = Query(None, description="Exact, case-sensitive match."),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List employees sorted by emp_code ascending."""
    q: Dict[str, Any] = {}
    if department is not None:
        q["department"] = department

    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    cursor = db.employees.find(q).sort("emp_code", ASCENDING).skip(skip).limit(page_size)

    items = [format_employee_doc(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Endpoints: Attendance Operations
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", response_model=AttendanceRecord, status_code=status.HTTP_201_CREATED, tags=["attendance"])
def punch_in(body: PunchInRequest):
    """Record a punch-in. Atomic insertion using compound unique index (emp_code, date)."""
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    punch_ms = body.punched_at if body.punched_at is not None else now_ms
    punch_ms = truncate_instant_ms(punch_ms)
    punch_dt = ms_to_utc_dt(punch_ms)

    att_date = calculate_attendance_date(punch_ms, emp["shift_start"], emp["shift_end"])
    late_minutes = compute_late_minutes(punch_dt, emp["shift_start"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status,
        "punch_in": punch_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_minutes,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }

    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="already punched in for that date")

    return format_attendance_doc(doc)


@app.post("/attendance/punch-out", response_model=AttendanceRecord, tags=["attendance"])
def punch_out(body: PunchOutRequest):
    """
    Record a punch-out. Closes the employee's most recent record whose punch_in <= punched_at.
    Two simultaneous punch-outs yield exactly one 200 and one 409 via atomic find_one_and_update.
    """
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    punch_out_ms = body.punched_at if body.punched_at is not None else now_ms
    punch_out_ms = truncate_instant_ms(punch_out_ms)
    punch_out_dt = ms_to_utc_dt(punch_out_ms)

    # Locate most recent record for employee with punch_in <= punch_out_dt (excluding null punch_ins)
    candidate = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": punch_out_dt}},
        sort=[("punch_in", DESCENDING)],
    )

    if not candidate:
        # Check if a punch-in exists that is after punch_out_dt (punched_at <= punch_in)
        future_candidate = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$gt": punch_out_dt}},
            sort=[("punch_in", ASCENDING)],
        )
        if future_candidate:
            raise HTTPException(
                status_code=422,
                detail="punched_at must be strictly after punch_in",
            )
        raise HTTPException(status_code=404, detail="No punch-in found")

    if candidate.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    punch_in_dt = candidate["punch_in"]
    if punch_in_dt.tzinfo is None:
        punch_in_dt = punch_in_dt.replace(tzinfo=UTC)
    punch_in_ms = int(punch_in_dt.timestamp() * 1000)

    # Validation: punched_at <= punch_in or more than 24 hours after it -> 422
    if punch_out_ms <= punch_in_ms or (punch_out_ms - punch_in_ms) > 24 * 3600 * 1000:
        raise HTTPException(
            status_code=422,
            detail="punched_at must be strictly after punch_in and within 24 hours",
        )

    work_hours = compute_work_hours(punch_in_dt, punch_out_dt)
    overtime_minutes = compute_overtime(
        punch_out_dt, emp["shift_start"], emp["shift_end"], candidate["date"]
    )
    half_day = compute_half_day(work_hours)

    updated = db.attendance_logs.find_one_and_update(
        {"_id": candidate["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hours,
                "overtime_minutes": overtime_minutes,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    return format_attendance_doc(updated)


@app.get("/attendance", response_model=AttendancePage, tags=["attendance"])
def list_attendance(
    emp_code: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List attendance records sorted by date descending, then emp_code ascending."""
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )

    items = [format_attendance_doc(doc) for doc in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}", response_model=AttendanceRecord, tags=["attendance"])
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
    body: RegularizeRequest = ...,
):
    """
    Regularize one day's record. Recompute derived fields and append to history.
    Concurrent edits safely return 409 if another patch modified history concurrently.
    """
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    old_status = record.get("status")
    old_punch_in = record.get("punch_in")
    old_punch_out = record.get("punch_out")
    old_work_hours = record.get("work_hours")
    old_late_minutes = record.get("late_minutes", 0) or 0
    old_overtime_minutes = record.get("overtime_minutes", 0) or 0
    old_half_day = bool(record.get("half_day", False))
    old_history = record.get("history", []) or []

    # Target values computation
    if body.status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                status_code=422,
                detail="punch_in and punch_out cannot be supplied when status is ABSENT or LEAVE",
            )
        target_status = body.status
        target_punch_in_dt = None
        target_punch_out_dt = None
        target_work_hours = None
        target_late_minutes = 0
        target_overtime_minutes = 0
        target_half_day = False
    else:
        target_status = body.status if body.status is not None else old_status
        if target_status in ("ABSENT", "LEAVE"):
            if body.punch_in is not None or body.punch_out is not None:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot supply punch times when status is ABSENT or LEAVE",
                )
            target_punch_in_dt = None
            target_punch_out_dt = None
            target_work_hours = None
            target_late_minutes = 0
            target_overtime_minutes = 0
            target_half_day = False
        else:
            # Presence status (PRESENT, WFH, ON_DUTY)
            if body.punch_in is not None:
                t_in_ms = truncate_instant_ms(body.punch_in)
            elif old_punch_in is not None:
                t_in_ms = dt_to_epoch_ms(old_punch_in)
            else:
                t_in_ms = None

            if t_in_ms is None:
                raise HTTPException(
                    status_code=422, detail="A presence status requires a punch_in"
                )

            # Punch_in must stay on the record's attendance date
            computed_date = calculate_attendance_date(t_in_ms, emp["shift_start"], emp["shift_end"])
            if computed_date != date:
                raise HTTPException(
                    status_code=422,
                    detail="punch_in must stay on the record's attendance date",
                )

            target_punch_in_dt = ms_to_utc_dt(t_in_ms)

            if body.punch_out is not None:
                t_out_ms = truncate_instant_ms(body.punch_out)
            elif old_punch_out is not None:
                t_out_ms = dt_to_epoch_ms(old_punch_out)
            else:
                t_out_ms = None

            if t_out_ms is not None:
                if t_out_ms <= t_in_ms or (t_out_ms - t_in_ms) > 24 * 3600 * 1000:
                    raise HTTPException(
                        status_code=422,
                        detail="punch_out must be after punch_in and within 24 hours",
                    )
                target_punch_out_dt = ms_to_utc_dt(t_out_ms)
                target_work_hours = compute_work_hours(target_punch_in_dt, target_punch_out_dt)
                target_overtime_minutes = compute_overtime(
                    target_punch_out_dt, emp["shift_start"], emp["shift_end"], date
                )
                target_half_day = compute_half_day(target_work_hours)
            else:
                target_punch_out_dt = None
                target_work_hours = None
                target_overtime_minutes = 0
                target_half_day = False

            target_late_minutes = compute_late_minutes(
                target_punch_in_dt, emp["shift_start"], date
            )

    norm_old_in = (
        old_punch_in.replace(tzinfo=UTC)
        if (isinstance(old_punch_in, datetime) and old_punch_in.tzinfo is None)
        else old_punch_in
    )
    norm_old_out = (
        old_punch_out.replace(tzinfo=UTC)
        if (isinstance(old_punch_out, datetime) and old_punch_out.tzinfo is None)
        else old_punch_out
    )

    # Determine changes
    changes_db = {}
    if target_status != old_status:
        changes_db["status"] = {"from": old_status, "to": target_status}
    if dt_to_epoch_ms(target_punch_in_dt) != dt_to_epoch_ms(norm_old_in):
        changes_db["punch_in"] = {"from": norm_old_in, "to": target_punch_in_dt}
    if dt_to_epoch_ms(target_punch_out_dt) != dt_to_epoch_ms(norm_old_out):
        changes_db["punch_out"] = {"from": norm_old_out, "to": target_punch_out_dt}
    if target_work_hours != old_work_hours:
        changes_db["work_hours"] = {"from": old_work_hours, "to": target_work_hours}
    if target_late_minutes != old_late_minutes:
        changes_db["late_minutes"] = {"from": old_late_minutes, "to": target_late_minutes}
    if target_overtime_minutes != old_overtime_minutes:
        changes_db["overtime_minutes"] = {"from": old_overtime_minutes, "to": target_overtime_minutes}
    if target_half_day != old_half_day:
        changes_db["half_day"] = {"from": old_half_day, "to": target_half_day}

    if not changes_db:
        raise HTTPException(status_code=422, detail="Request changes nothing")

    now_dt = datetime.now(UTC).replace(microsecond=0)
    history_entry = {
        "at": now_dt,
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes_db,
    }

    # Safe atomic update checking current history length to prevent lost audit entries
    condition = {
        "_id": record["_id"],
        f"history.{len(old_history)}": {"$exists": False},
    }
    if len(old_history) > 0:
        condition[f"history.{len(old_history) - 1}"] = {"$exists": True}

    updated = db.attendance_logs.find_one_and_update(
        condition,
        {
            "$set": {
                "status": target_status,
                "punch_in": target_punch_in_dt,
                "punch_out": target_punch_out_dt,
                "work_hours": target_work_hours,
                "late_minutes": target_late_minutes,
                "overtime_minutes": target_overtime_minutes,
                "half_day": target_half_day,
            },
            "$push": {"history": history_entry},
        },
        return_document=ReturnDocument.AFTER,
    )

    if not updated:
        raise HTTPException(
            status_code=409,
            detail="Concurrent modification detected. Please re-read and retry.",
        )

    return format_attendance_doc(updated)


# --------------------------------------------------------------------------- #
# Aggregation Query / Pipeline Shared Builders
# --------------------------------------------------------------------------- #
def get_month_date_range(month_str: str) -> tuple[str, str]:
    year, m = map(int, month_str.split("-"))
    next_m = m + 1 if m < 12 else 1
    next_y = year if m < 12 else year + 1
    first_day = f"{month_str}-01"
    last_day = (datetime(next_y, next_m, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
    return first_day, last_day


def count_working_days(joined_on_str: str, month_first_str: str, month_last_str: str) -> int:
    """R7: Monday-Friday working days, counted from joined_on for mid-month joiners."""
    first_dt = date.fromisoformat(month_first_str)
    last_dt = date.fromisoformat(month_last_str)
    joined_dt = date.fromisoformat(joined_on_str)

    start_dt = max(first_dt, joined_dt)
    if start_dt > last_dt:
        return 0

    working_days = 0
    cur = start_dt
    while cur <= last_dt:
        if cur.weekday() < 5:  # Mon=0, Fri=4
            working_days += 1
        cur += timedelta(days=1)
    return working_days


def build_employee_monthly_pipeline(emp_code: str, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": date_from, "$lte": date_to}}},
        {
            "$group": {
                "_id": None,
                "present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {
                                        "$lte": [
                                            {"$isoDayOfWeek": {"$dateFromString": {"dateString": "$date"}}},
                                            5,
                                        ]
                                    },
                                ]
                            },
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"},
            }
        },
    ]


def build_department_summary_pipeline(
    date_from: str, date_to: str, department: Optional[str] = None
) -> List[Dict[str, Any]]:
    match_stage: Dict[str, Any] = {"joined_on": {"$lte": date_to}}
    if department:
        match_stage["department"] = department

    return [
        {"$match": match_stage},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$gte": ["$date", date_from]},
                                    {"$lte": ["$date", date_to]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {
            "$project": {
                "department": 1,
                "emp_present_days": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": {
                                        "$and": [
                                            {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                            {
                                                "$lte": [
                                                    {
                                                        "$isoDayOfWeek": {
                                                            "$dateFromString": {
                                                                "dateString": "$$l.date"
                                                            }
                                                        }
                                                    },
                                                    5,
                                                ]
                                            },
                                        ]
                                    },
                                }
                            },
                            "as": "p",
                            "in": {"$cond": [{"$eq": ["$$p.half_day", True]}, 0.5, 1.0]},
                        }
                    }
                },
                "emp_leave_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "LEAVE"]},
                        }
                    }
                },
                "emp_on_duty_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$eq": ["$$l.status", "ON_DUTY"]},
                        }
                    }
                },
                "emp_late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": ["$$l.late_minutes", 0]},
                        }
                    }
                },
                "emp_total_late_minutes": {"$sum": "$logs.late_minutes"},
                "emp_work_hours_sum": {
                    "$sum": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$logs",
                                    "as": "l",
                                    "cond": {
                                        "$and": [
                                            {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                            {"$ne": ["$$l.work_hours", None]},
                                        ]
                                    },
                                }
                            },
                            "as": "w",
                            "in": "$$w.work_hours",
                        }
                    }
                },
                "emp_work_hours_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {
                                "$and": [
                                    {"$in": ["$$l.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$ne": ["$$l.work_hours", None]},
                                ]
                            },
                        }
                    }
                },
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$emp_present_days"},
                "leave_count": {"$sum": "$emp_leave_count"},
                "on_duty_count": {"$sum": "$emp_on_duty_count"},
                "late_count": {"$sum": "$emp_late_count"},
                "total_late_minutes": {"$sum": "$emp_total_late_minutes"},
                "work_hours_sum": {"$sum": "$emp_work_hours_sum"},
                "work_hours_count": {"$sum": "$emp_work_hours_count"},
            }
        },
        {"$sort": {"_id": ASCENDING}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": 1,
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        {"$divide": ["$work_hours_sum", "$work_hours_count"]},
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
    ]


def build_late_leaderboard_pipeline(
    date_from: str, date_to: str, limit: int = 10, department: Optional[str] = None
) -> List[Dict[str, Any]]:
    pipe: List[Dict[str, Any]] = [
        {
            "$match": {
                "date": {"$gte": date_from, "$lte": date_to},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]

    if department:
        pipe.append({"$match": {"emp.department": department}})

    pipe.extend([
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": DESCENDING},
                "output": {
                    "rank": {
                        "$rank": {}
                    }
                },
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": DESCENDING, "_id": ASCENDING}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ])
    return pipe


def build_department_trend_pipeline(
    department: str, from_str: str, to_str: str
) -> List[Dict[str, Any]]:
    from_dt = datetime.fromisoformat(from_str).replace(tzinfo=UTC)
    to_dt = (datetime.fromisoformat(to_str) + timedelta(days=1)).replace(tzinfo=UTC)

    return [
        {"$match": {"date": {"$gte": from_str, "$lte": to_str}}},
        {
            "$lookup": {
                "from": "employees",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
        {"$match": {"emp.department": department}},
        {
            "$group": {
                "_id": "$date",
                "present_count": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "dt": {"$dateFromString": {"dateString": "$_id"}},
                "present_count": 1,
                "late_count": 1,
            }
        },
        {
            "$densify": {
                "field": "dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [from_dt, to_dt],
                },
            }
        },
        {
            "$project": {
                "dt": 1,
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$dt"}},
                "present_count": {"$ifNull": ["$present_count", 0.0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "is_working_day": {"$lte": [{"$isoDayOfWeek": "$dt"}, 5]},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "let": {"cur_date": "$date"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$department", department]},
                                    {"$lte": ["$joined_on", "$$cur_date"]},
                                ]
                            }
                        }
                    },
                    {"$count": "count"},
                ],
                "as": "hc",
            }
        },
        {
            "$addFields": {
                "headcount": {"$ifNull": [{"$first": "$hc.count"}, 0]},
            }
        },
        {
            "$addFields": {
                "attendance_rate": {
                    "$cond": [
                        {
                            "$and": [
                                {"$eq": ["$is_working_day", True]},
                                {"$gt": ["$headcount", 0]},
                            ]
                        },
                        {"$divide": ["$present_count", "$headcount"]},
                        None,
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "moving_avg_7d": {
                        "$avg": "$attendance_rate",
                        "window": {
                            "documents": [-6, "current"],
                        },
                    }
                },
            }
        },
        {"$sort": {"date": 1}},
    ]


# --------------------------------------------------------------------------- #
# Endpoints: Analytics
# --------------------------------------------------------------------------- #
@app.get(
    "/analytics/employees/{emp_code}/monthly",
    response_model=EmployeeMonthly,
    tags=["analytics"],
)
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    """Monthly attendance summary for one employee via MongoDB aggregation."""
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    date_from, date_to = get_month_date_range(month)
    working_days = count_working_days(emp["joined_on"], date_from, date_to)

    pipeline = build_employee_monthly_pipeline(emp_code, date_from, date_to)
    agg_res = list(db.attendance_logs.aggregate(pipeline))

    if agg_res:
        row = agg_res[0]
        present_days = float(row.get("present_days", 0.0))
        leave_days = int(row.get("leave_days", 0))
        late_count = int(row.get("late_count", 0))
        total_late = int(row.get("total_late_minutes", 0))
        total_ot = int(row.get("total_overtime_minutes", 0))
    else:
        present_days = 0.0
        leave_days = 0
        late_count = 0
        total_late = 0
        total_ot = 0

    if working_days > 0:
        attendance_pct = round_half_up((present_days / working_days) * 100.0, 2)
    else:
        attendance_pct = None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late,
        "total_overtime_minutes": total_ot,
        "attendance_pct": attendance_pct,
    }


@app.get(
    "/analytics/departments/summary",
    response_model=DepartmentSummary,
    tags=["analytics"],
)
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
):
    """Department summary for a month via MongoDB cross-collection aggregation."""
    date_from, date_to = get_month_date_range(month)
    pipeline = build_department_summary_pipeline(date_from, date_to, department)

    results = list(db.employees.aggregate(pipeline))
    items = []
    for r in results:
        avg_wh = r.get("avg_work_hours")
        if avg_wh is not None:
            avg_wh = round_half_up(float(avg_wh), 2)

        items.append({
            "department": r["department"],
            "headcount": int(r["headcount"]),
            "present_days": float(r.get("present_days", 0.0)),
            "avg_work_hours": avg_wh,
            "late_count": int(r.get("late_count", 0)),
            "total_late_minutes": int(r.get("total_late_minutes", 0)),
            "leave_count": int(r.get("leave_count", 0)),
            "on_duty_count": int(r.get("on_duty_count", 0)),
        })

    return {"month": month, "items": items}


@app.get(
    "/analytics/leaderboard/late",
    response_model=Leaderboard,
    tags=["analytics"],
)
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    """Late-comers leaderboard for a month using MongoDB window ranking ($rank)."""
    date_from, date_to = get_month_date_range(month)
    pipeline = build_late_leaderboard_pipeline(date_from, date_to, limit, department)

    results = list(db.attendance_logs.aggregate(pipeline))
    items = [
        {
            "rank": int(r["rank"]),
            "emp_code": r["emp_code"],
            "name": r["name"],
            "department": r["department"],
            "total_late_minutes": int(r["total_late_minutes"]),
            "late_count": int(r["late_count"]),
        }
        for r in results
    ]

    return {"month": month, "items": items}


@app.get(
    "/analytics/departments/{department}/trend",
    response_model=Trend,
    tags=["analytics"],
)
def department_trend(
    department: str = Path(...),
    from_date: str = Query(..., alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: str = Query(..., alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
):
    """Daily attendance trend with database-side gap filling ($densify) and 7-day moving average."""
    # Validate department exists
    if not db.employees.find_one({"department": department}):
        raise HTTPException(status_code=404, detail="Department not found")

    d_from = date.fromisoformat(from_date)
    d_to = date.fromisoformat(to_date)
    if d_to < d_from:
        raise HTTPException(status_code=422, detail="to date cannot be earlier than from date")

    day_span = (d_to - d_from).days + 1
    if day_span > 92:
        raise HTTPException(status_code=422, detail="Date range exceeds 92 days")

    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    results = list(db.attendance_logs.aggregate(pipeline))

    items = []
    for r in results:
        rate = r.get("attendance_rate")
        if rate is not None:
            rate = round_half_up(float(rate), 4)

        mov_avg = r.get("moving_avg_7d")
        if mov_avg is not None:
            mov_avg = round_half_up(float(mov_avg), 4)

        items.append({
            "date": r["date"],
            "is_working_day": bool(r["is_working_day"]),
            "headcount": int(r["headcount"]),
            "present_count": float(r["present_count"]),
            "late_count": int(r["late_count"]),
            "attendance_rate": rate,
            "moving_avg_7d": mov_avg,
        })

    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Endpoints: Admin Explain
# --------------------------------------------------------------------------- #
@app.get("/admin/explain/{endpoint}", tags=["admin"])
def explain_endpoint(
    endpoint: str = Path(
        ...,
        pattern=r"^(attendance_list|employee_monthly|department_summary|late_leaderboard|department_trend)$",
    ),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: Optional[int] = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    date_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    status: Optional[str] = Query(None, pattern=r"^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$"),
    from_date: Optional[str] = Query(None, alias="from", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    to_date: Optional[str] = Query(None, alias="to", pattern=r"^\d{4}-\d{2}-\d{2}$"),
    page: Optional[int] = Query(1, ge=1),
    page_size: Optional[int] = Query(20, ge=1, le=100),
):
    """
    Returns raw MongoDB explain output with executionStats verbosity for an endpoint's main query.
    Validates required parameters for each endpoint and rejects missing parameters with 422.
    """
    if endpoint == "attendance_list":
        if date_from and date_to and date_from > date_to:
            raise HTTPException(status_code=422, detail="date_from must be <= date_to")

        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status

        p = page or 1
        ps = page_size or 20
        skip = (p - 1) * ps

        explain_doc = db.command(
            "explain",
            {
                "find": "attendance_logs",
                "filter": q,
                "sort": {"date": -1, "emp_code": 1},
                "skip": skip,
                "limit": ps,
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_doc,
        }

    elif endpoint == "employee_monthly":
        if not emp_code:
            raise HTTPException(status_code=422, detail="emp_code is required for employee_monthly")
        if not month:
            raise HTTPException(status_code=422, detail="month is required for employee_monthly")

        d_from, d_to = get_month_date_range(month)
        pipeline = build_employee_monthly_pipeline(emp_code, d_from, d_to)

        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_doc,
        }

    elif endpoint == "department_summary":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for department_summary")

        d_from, d_to = get_month_date_range(month)
        pipeline = build_department_summary_pipeline(d_from, d_to, department)

        explain_doc = db.command(
            "explain",
            {
                "aggregate": "employees",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "employees",
            "explain": explain_doc,
        }

    elif endpoint == "late_leaderboard":
        if not month:
            raise HTTPException(status_code=422, detail="month is required for late_leaderboard")

        d_from, d_to = get_month_date_range(month)
        pipeline = build_late_leaderboard_pipeline(d_from, d_to, limit or 10, department)

        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_doc,
        }

    elif endpoint == "department_trend":
        if not department:
            raise HTTPException(status_code=422, detail="department is required for department_trend")
        if not from_date or not to_date:
            raise HTTPException(status_code=422, detail="from and to are required for department_trend")

        d_from = date.fromisoformat(from_date)
        d_to = date.fromisoformat(to_date)
        if d_to < d_from:
            raise HTTPException(status_code=422, detail="to date cannot be earlier than from date")
        if (d_to - d_from).days + 1 > 92:
            raise HTTPException(status_code=422, detail="Date range exceeds 92 days")

        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        explain_doc = db.command(
            "explain",
            {
                "aggregate": "attendance_logs",
                "pipeline": pipeline,
                "cursor": {},
            },
            verbosity="executionStats",
        )
        return {
            "endpoint": endpoint,
            "collection": "attendance_logs",
            "explain": explain_doc,
        }

    raise HTTPException(status_code=422, detail=f"Unsupported endpoint: {endpoint}")
