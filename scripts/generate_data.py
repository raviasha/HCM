"""
Synthetic data generator for HCM AI Insights prototype.

Generates **structurally different** datasets for 3 companies to
demonstrate the app's dynamic analysis capabilities:

  1. NovaTech Solutions   – Tech · CSV target = Attrition (binary Yes/No)
  2. Meridian Retail Group – Retail · CSV target = EngagementScore (numeric 1-10)
  3. Pinnacle Healthcare   – Healthcare · CSV target = BurnoutIndex (numeric 1-10)

Each company also uses different CSV column names and different JSON
feedback schemas so the AI must adapt to every upload.

Usage:
    python scripts/generate_data.py                # Generate all 3 companies
    python scripts/generate_data.py --company novatech
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

# ── Output directory ─────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic"


# ═════════════════════════════════════════════════════════════════════
#  COMPANY 1: NovaTech Solutions — Binary target: Attrition (Yes/No)
# ═════════════════════════════════════════════════════════════════════

NOVATECH = {
    "name": "NovaTech Solutions",
    "slug": "novatech",
    "employee_count": 1500,
    "attrition_base_rate": 0.14,
    "departments": {
        "Engineering":  {"weight": 0.35, "boost": 0.10},
        "Product":      {"weight": 0.12, "boost": 0.02},
        "Sales":        {"weight": 0.15, "boost": 0.03},
        "Marketing":    {"weight": 0.10, "boost": 0.00},
        "HR":           {"weight": 0.08, "boost": -0.02},
        "Finance":      {"weight": 0.08, "boost": -0.03},
        "Operations":   {"weight": 0.12, "boost": 0.01},
    },
    "roles": {
        "Engineering":  ["Software Engineer", "Senior Engineer", "Staff Engineer", "Engineering Manager", "DevOps Engineer", "QA Engineer", "Data Engineer"],
        "Product":      ["Product Manager", "UX Designer", "Product Analyst", "UI Designer"],
        "Sales":        ["Account Executive", "Sales Rep", "Sales Manager", "BDR"],
        "Marketing":    ["Marketing Analyst", "Content Strategist", "Growth Manager", "Brand Manager"],
        "HR":           ["HR Generalist", "Recruiter", "HR Manager", "Compensation Analyst"],
        "Finance":      ["Financial Analyst", "Controller", "Accountant", "FP&A Manager"],
        "Operations":   ["Operations Manager", "IT Support", "Facilities Coordinator", "Project Manager"],
    },
    "overtime_rate": 0.40,
    "feedback_positive": [
        "I love the technical challenges and innovative projects we work on.",
        "The team culture in engineering is collaborative and supportive.",
        "Great learning opportunities — I've grown so much technically here.",
        "Flexible remote work policy is a huge perk.",
        "Access to cutting-edge tools and technology is excellent.",
        "My manager truly cares about my professional development.",
        "The hackathon events are great for team bonding and innovation.",
        "I appreciate the transparent communication from leadership.",
    ],
    "feedback_negative": [
        "The overtime culture is unsustainable. I'm burning out working 60+ hour weeks.",
        "Haven't been promoted in 3 years despite consistently strong reviews.",
        "There's no clear career path for senior individual contributors.",
        "On-call rotations are killing my work-life balance.",
        "Middle management seems disconnected from what engineers actually need.",
        "Technical debt keeps piling up but we never get time to address it.",
        "I feel like just a resource, not a person. Sprint after sprint with no breaks.",
        "The promotion process is opaque and seems based on politics not merit.",
        "We keep losing our best engineers to competitors offering better comp.",
        "Constant context switching between projects makes deep work impossible.",
        "I've raised concerns about burnout multiple times but nothing changes.",
        "The expectation to be always available on Slack is exhausting.",
    ],
    "feedback_neutral": [
        "The work is interesting but the pace is intense.",
        "Benefits are standard for the industry, nothing exceptional.",
        "Office space is fine, nothing to complain about.",
        "Communication could be better between teams.",
        "Training budget exists but it's hard to find time to use it.",
    ],
}


def _novatech_employee(eid: int, profile: dict) -> dict:
    depts = list(profile["departments"].keys())
    weights = [profile["departments"][d]["weight"] for d in depts]
    dept = random.choices(depts, weights=weights, k=1)[0]
    role = random.choice(profile["roles"][dept])
    level = random.choices([1, 2, 3, 4, 5], weights=[25, 30, 25, 15, 5], k=1)[0]
    age = random.randint(22, 62)
    years_at = random.randint(0, min(age - 22, 30))
    years_role = random.randint(0, years_at)
    years_promo = random.randint(0, years_at)
    income = int({1: 3000, 2: 5000, 3: 7500, 4: 10000, 5: 14000}[level] * random.uniform(0.8, 1.4))
    overtime = "Yes" if random.random() < profile["overtime_rate"] else "No"
    satisfaction = random.randint(1, 4)
    engagement = round(random.uniform(3.0, 9.5), 1)

    # Attrition probability
    base = profile["attrition_base_rate"] + profile["departments"][dept]["boost"]
    if overtime == "Yes":
        base *= 1.8
    if years_promo >= 4:
        base *= 1.5
    if satisfaction <= 2:
        base *= 1.4
    if income < 4000:
        base *= 1.2
    if age < 30:
        base *= 1.15
    elif age > 50:
        base *= 0.85
    base = min(base, 0.85)

    return {
        "EmployeeID": eid,
        "Age": age,
        "Gender": random.choice(["Male", "Female", "Non-Binary"]),
        "MaritalStatus": random.choice(["Single", "Married", "Divorced"]),
        "Department": dept,
        "JobRole": role,
        "JobLevel": level,
        "MonthlyIncome": income,
        "YearsAtCompany": years_at,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promo,
        "TotalWorkingYears": random.randint(years_at, age - 18),
        "OverTime": overtime,
        "DistanceFromHome": random.randint(1, 50),
        "Education": random.randint(1, 5),
        "PerformanceRating": random.choices([1, 2, 3, 4], weights=[5, 15, 50, 30], k=1)[0],
        "JobSatisfaction": satisfaction,
        "EnvironmentSatisfaction": random.randint(1, 4),
        "WorkLifeBalance": random.randint(1, 4),
        "RelationshipSatisfaction": random.randint(1, 4),
        "NumCompaniesWorked": random.randint(0, 9),
        "TrainingTimesLastYear": random.randint(0, 6),
        "StockOptionLevel": random.randint(0, 3),
        "PercentSalaryHike": random.randint(11, 25),
        "EngagementScore": engagement,
        "Attrition": "Yes" if random.random() < base else "No",
    }


def _novatech_feedback(employees: list[dict], profile: dict) -> list[dict]:
    """NovaTech JSON schema: feedback_id, employee_id, feedback_type, date,
    question_prompt, response_text, department"""
    themes = (profile["feedback_positive"], profile["feedback_neutral"], profile["feedback_negative"])
    prompts = {
        "exit_interview": [
            "What is your primary reason for leaving?",
            "What could we have done differently to retain you?",
            "How would you describe the work culture?",
        ],
        "pulse_survey": [
            "How are you feeling about your work this week?",
            "Is there anything impacting your productivity right now?",
        ],
        "engagement_survey": [
            "What do you value most about working here?",
            "What is one thing we could improve about your work experience?",
            "How supported do you feel by your direct manager?",
        ],
        "open_comment": [
            "Share any feedback about your experience at the company.",
            "Is there anything you'd like leadership to know?",
        ],
    }
    base_date = datetime(2025, 1, 1)
    feedback, fid = [], 1

    attrited = [e for e in employees if e["Attrition"] == "Yes"]
    active = [e for e in employees if e["Attrition"] == "No"]

    for emp in attrited:
        sent = random.random()
        text = random.choice(themes[2]) if sent < 0.80 else random.choice(themes[1]) if sent < 0.90 else random.choice(themes[0])
        feedback.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": "exit_interview",
            "date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "question_prompt": random.choice(prompts["exit_interview"]),
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1

    for emp in random.sample(active, min(len(active), 350)):
        ft = random.choice(["pulse_survey", "engagement_survey"])
        r = random.random()
        text = random.choice(themes[0]) if r < 0.35 else random.choice(themes[1]) if r < 0.60 else random.choice(themes[2])
        feedback.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": ft,
            "date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "question_prompt": random.choice(prompts[ft]),
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1

    for _ in range(80):
        emp = random.choice(employees)
        r = random.random()
        text = random.choice(themes[0]) if r < 0.25 else random.choice(themes[1]) if r < 0.45 else random.choice(themes[2])
        feedback.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": "open_comment",
            "date": (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
            "question_prompt": random.choice(prompts["open_comment"]),
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1
    return feedback


# ═════════════════════════════════════════════════════════════════════
#  COMPANY 2: Meridian Retail — Numeric target: EngagementScore (1-10)
#  Completely different CSV columns + JSON schema
# ═════════════════════════════════════════════════════════════════════

MERIDIAN = {
    "name": "Meridian Retail Group",
    "slug": "meridian",
    "employee_count": 1800,
    "departments": {
        "Sales Floor":   {"weight": 0.30},
        "Warehouse":     {"weight": 0.20},
        "Customer Service": {"weight": 0.15},
        "Marketing":     {"weight": 0.10},
        "HR":            {"weight": 0.07},
        "Finance":       {"weight": 0.06},
        "Supply Chain":  {"weight": 0.07},
        "IT":            {"weight": 0.05},
    },
    "roles": {
        "Sales Floor":      ["Store Associate", "Store Manager", "Department Lead", "Cashier", "Visual Merchandiser"],
        "Warehouse":        ["Warehouse Worker", "Forklift Operator", "Inventory Analyst", "Shift Supervisor", "Receiving Clerk"],
        "Customer Service": ["CS Representative", "CS Manager", "Returns Specialist", "Call Center Agent"],
        "Marketing":        ["Marketing Manager", "Digital Marketing Specialist", "Brand Analyst", "Social Media Coordinator"],
        "HR":               ["HR Generalist", "Recruiter", "Training Coordinator", "HR Manager"],
        "Finance":          ["Accountant", "Payroll Specialist", "Financial Analyst", "Controller"],
        "Supply Chain":     ["Procurement Analyst", "Demand Planner", "Buyer", "Logistics Coordinator"],
        "IT":               ["IT Support", "Systems Administrator", "POS Technician", "IT Manager"],
    },
    "regions": ["Northeast", "Southeast", "Midwest", "Southwest", "West Coast"],
    "feedback_positive": [
        "The team on the floor is like family, we look out for each other.",
        "I enjoy working with customers and helping them find what they need.",
        "Employee discount is a great perk — I save a lot every month.",
        "My store manager is understanding about scheduling needs.",
        "I like the fast-paced environment and no two days are the same.",
        "The company has good values around community involvement.",
        "Our regional manager actually listens when we raise concerns.",
        "The new POS system makes checkout much smoother. Good investment.",
    ],
    "feedback_negative": [
        "The pay is not enough to live on. I need a second job just to cover rent.",
        "Scheduling is unpredictable — I never know my hours more than a week out.",
        "Standing for 10 hours with a 30-minute break is physically exhausting.",
        "Holiday seasons are brutal. We're severely understaffed during peak times.",
        "There's no career growth path from store associate to anything meaningful.",
        "Benefits are minimal for part-time workers, which is most of us.",
        "Commission structure keeps changing and it's always in the company's favor.",
        "I've been asking for full-time status for months with no response.",
        "Corporate doesn't understand what it's like on the ground in stores.",
        "Turnover is so high I'm constantly training new people instead of selling.",
        "The AC in our store has been broken for weeks and management doesn't care.",
        "I can't afford the health insurance they offer on my salary.",
    ],
    "feedback_neutral": [
        "The job is straightforward, no surprises.",
        "Some weeks are better than others depending on the schedule.",
        "Training was adequate but could be more thorough.",
        "The break room could use some updating.",
        "Product knowledge sessions are somewhat helpful.",
    ],
}


def _meridian_employee(eid: int, profile: dict) -> dict:
    depts = list(profile["departments"].keys())
    weights = [profile["departments"][d]["weight"] for d in depts]
    dept = random.choices(depts, weights=weights, k=1)[0]
    role = random.choice(profile["roles"][dept])
    region = random.choice(profile["regions"])
    age = random.randint(18, 65)
    tenure_months = random.randint(1, max(1, min((age - 18) * 12, 240)))
    shift = random.choice(["Morning", "Afternoon", "Evening", "Rotating"])
    weekly_hours = random.choices(
        [20, 25, 30, 35, 40, 45, 50],
        weights=[10, 15, 15, 15, 25, 12, 8],
        k=1,
    )[0]
    employment = "Full-Time" if weekly_hours >= 35 else "Part-Time"
    hourly_wage = round(random.uniform(12.0, 32.0), 2)
    if dept in ("IT", "Finance", "Marketing"):
        hourly_wage = round(random.uniform(22.0, 45.0), 2)
    cust_sat = round(random.uniform(2.5, 5.0), 1) if dept in ("Sales Floor", "Customer Service") else None
    manager_rating = random.randint(1, 5)
    training_hours = random.randint(0, 40)
    commute_minutes = random.randint(5, 90)

    # EngagementScore influenced by factors
    eng = 5.0
    if hourly_wage > 25:
        eng += random.uniform(0.5, 1.5)
    elif hourly_wage < 15:
        eng -= random.uniform(0.5, 1.5)
    if weekly_hours > 45:
        eng -= random.uniform(0.5, 1.0)
    if manager_rating >= 4:
        eng += random.uniform(0.3, 1.0)
    elif manager_rating <= 2:
        eng -= random.uniform(0.5, 1.5)
    if shift == "Rotating":
        eng -= random.uniform(0.3, 0.8)
    if tenure_months > 60:
        eng += random.uniform(0.2, 0.5)
    eng += random.uniform(-1.0, 1.0)
    eng = round(max(1.0, min(10.0, eng)), 1)

    row = {
        "StaffID": eid,
        "Age": age,
        "Gender": random.choice(["Male", "Female", "Non-Binary"]),
        "Department": dept,
        "Role": role,
        "StoreRegion": region,
        "EmploymentType": employment,
        "ShiftType": shift,
        "WeeklyHours": weekly_hours,
        "HourlyWage": hourly_wage,
        "TenureMonths": tenure_months,
        "ManagerRating": manager_rating,
        "TrainingHoursLastYear": training_hours,
        "CommuteMinutes": commute_minutes,
        "EngagementScore": eng,
    }
    if cust_sat is not None:
        row["CustomerSatRating"] = cust_sat
    else:
        row["CustomerSatRating"] = ""
    return row


def _meridian_feedback(employees: list[dict], profile: dict) -> list[dict]:
    """Meridian JSON schema: id, staff_id, survey_type, submitted_at,
    question, answer, store_region, department"""
    themes = (profile["feedback_positive"], profile["feedback_neutral"], profile["feedback_negative"])
    questions = {
        "quarterly_pulse": [
            "How engaged do you feel at work right now?",
            "Do you feel recognized for your contributions?",
            "How would you rate your work-life balance this quarter?",
        ],
        "annual_review": [
            "What motivates you to come to work each day?",
            "What is the biggest challenge you face in your role?",
            "How could your manager better support you?",
            "Would you recommend Meridian as an employer? Why?",
        ],
        "suggestion_box": [
            "Any suggestions for improving the workplace?",
            "What would make your job easier or more enjoyable?",
        ],
    }
    base_date = datetime(2025, 1, 1)
    fb, fid = [], 1

    # Low-engagement employees get more negative feedback
    low_eng = [e for e in employees if e["EngagementScore"] <= 4.0]
    high_eng = [e for e in employees if e["EngagementScore"] >= 7.0]
    mid_eng = [e for e in employees if 4.0 < e["EngagementScore"] < 7.0]

    for emp in random.sample(low_eng, min(len(low_eng), 250)):
        st = random.choice(["quarterly_pulse", "annual_review"])
        text = random.choice(themes[2]) if random.random() < 0.75 else random.choice(themes[1])
        fb.append({
            "id": fid, "staff_id": emp["StaffID"],
            "survey_type": st,
            "submitted_at": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "question": random.choice(questions[st]),
            "answer": text,
            "store_region": emp["StoreRegion"],
            "department": emp["Department"],
        })
        fid += 1

    for emp in random.sample(high_eng, min(len(high_eng), 200)):
        st = random.choice(["quarterly_pulse", "annual_review"])
        text = random.choice(themes[0]) if random.random() < 0.70 else random.choice(themes[1])
        fb.append({
            "id": fid, "staff_id": emp["StaffID"],
            "survey_type": st,
            "submitted_at": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "question": random.choice(questions[st]),
            "answer": text,
            "store_region": emp["StoreRegion"],
            "department": emp["Department"],
        })
        fid += 1

    for emp in random.sample(mid_eng, min(len(mid_eng), 200)):
        st = "quarterly_pulse"
        r = random.random()
        text = random.choice(themes[0]) if r < 0.30 else random.choice(themes[1]) if r < 0.55 else random.choice(themes[2])
        fb.append({
            "id": fid, "staff_id": emp["StaffID"],
            "survey_type": st,
            "submitted_at": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "question": random.choice(questions[st]),
            "answer": text,
            "store_region": emp["StoreRegion"],
            "department": emp["Department"],
        })
        fid += 1

    for _ in range(100):
        emp = random.choice(employees)
        text = random.choice(themes[0] + themes[1] + themes[2])
        fb.append({
            "id": fid, "staff_id": emp["StaffID"],
            "survey_type": "suggestion_box",
            "submitted_at": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "question": random.choice(questions["suggestion_box"]),
            "answer": text,
            "store_region": emp["StoreRegion"],
            "department": emp["Department"],
        })
        fid += 1
    return fb


# ═════════════════════════════════════════════════════════════════════
#  COMPANY 3: Pinnacle Healthcare — Numeric target: BurnoutIndex (1-10)
#  Completely different columns again
# ═════════════════════════════════════════════════════════════════════

PINNACLE = {
    "name": "Pinnacle Healthcare",
    "slug": "pinnacle",
    "employee_count": 1600,
    "units": {
        "Emergency":     {"weight": 0.18},
        "Surgery":       {"weight": 0.12},
        "ICU":           {"weight": 0.10},
        "General Ward":  {"weight": 0.15},
        "Outpatient":    {"weight": 0.12},
        "Administration": {"weight": 0.10},
        "Lab & Diagnostics": {"weight": 0.08},
        "Pharmacy":      {"weight": 0.05},
        "Facilities":    {"weight": 0.10},
    },
    "roles": {
        "Emergency":     ["ER Physician", "ER Nurse", "Paramedic", "Triage Nurse", "ER Technician"],
        "Surgery":       ["Surgeon", "Surgical Nurse", "Anesthesiologist", "Surgical Technician", "OR Coordinator"],
        "ICU":           ["ICU Nurse", "Intensivist", "Respiratory Therapist", "ICU Technician"],
        "General Ward":  ["Registered Nurse", "Nurse Practitioner", "CNA", "Charge Nurse", "Ward Clerk"],
        "Outpatient":    ["Physician", "Medical Assistant", "Scheduler", "Nurse Practitioner"],
        "Administration": ["Admin Assistant", "Office Manager", "Medical Records Clerk", "Billing Specialist", "HR Coordinator"],
        "Lab & Diagnostics": ["Lab Technician", "Radiologist", "Phlebotomist", "Pathologist", "MRI Technician"],
        "Pharmacy":      ["Pharmacist", "Pharmacy Technician", "Clinical Pharmacist"],
        "Facilities":    ["Maintenance Technician", "Environmental Services", "Safety Officer", "Facilities Manager"],
    },
    "feedback_positive": [
        "I find deep meaning in helping patients recover and improve their lives.",
        "My colleagues are dedicated professionals who inspire me daily.",
        "The hospital's mission to serve the community keeps me motivated.",
        "Leadership has been investing in new medical equipment which is great.",
        "I appreciate the continuing education reimbursement program.",
        "The patient outcomes we achieve together make the hard days worth it.",
        "Our new scheduling system is much fairer than the old one.",
        "I'm proud to work at a hospital that genuinely puts patients first.",
    ],
    "feedback_negative": [
        "I'm working double shifts multiple times a week. Patient safety is at risk when staff are exhausted.",
        "Management keeps cutting headcount while patient volume increases.",
        "The emotional toll of this work is immense and there's zero mental health support for staff.",
        "My manager hasn't checked in on my wellbeing once in 6 months.",
        "We're asked to do more with less every quarter. Morale is at an all-time low.",
        "The bureaucracy around simple decisions is suffocating.",
        "I love patient care but the administrative burden is crushing.",
        "Night shifts with no additional compensation is demoralizing.",
        "We've lost 5 nurses this quarter alone and there's no plan to replace them.",
        "The lack of work-life balance in healthcare is unsustainable long-term.",
        "I've been passed over for the charge nurse role twice with no explanation.",
        "Parking costs, mandatory scrubs we buy ourselves — it all adds up on nurse pay.",
    ],
    "feedback_neutral": [
        "The cafeteria food is decent for a hospital.",
        "Orientation was comprehensive but the first week was overwhelming.",
        "IT systems are functional but not intuitive.",
        "The break rooms could use some renovation.",
        "Scheduling is fair most of the time.",
    ],
}


def _pinnacle_employee(eid: int, profile: dict) -> dict:
    units = list(profile["units"].keys())
    weights = [profile["units"][u]["weight"] for u in units]
    unit = random.choices(units, weights=weights, k=1)[0]
    role = random.choice(profile["roles"][unit])
    age = random.randint(22, 68)
    years_exp = random.randint(0, min(age - 22, 40))
    shift = random.choice(["Day", "Night", "Swing", "Rotating 12h"])
    weekly_patient_load = 0
    if unit in ("Emergency", "ICU", "Surgery", "General Ward"):
        weekly_patient_load = random.randint(8, 45)
    elif unit in ("Outpatient", "Lab & Diagnostics", "Pharmacy"):
        weekly_patient_load = random.randint(15, 80)

    cert_level = random.choice(["Entry", "Intermediate", "Advanced", "Specialist"])
    annual_salary = random.randint(32000, 180000)
    if unit == "Administration":
        annual_salary = random.randint(35000, 85000)
    elif unit == "Facilities":
        annual_salary = random.randint(30000, 60000)
    elif role in ("Surgeon", "Anesthesiologist", "Intensivist", "Radiologist", "Pathologist"):
        annual_salary = random.randint(150000, 350000)

    overtime_hours = random.choices(
        [0, 5, 10, 15, 20, 30],
        weights=[20, 25, 20, 15, 12, 8],
        k=1,
    )[0]
    has_dependents = random.choice(["Yes", "No"])
    commute = random.randint(5, 75)
    mgr_support = random.randint(1, 5)
    peer_support = random.randint(1, 5)

    # BurnoutIndex: higher = more burned out
    burn = 5.0
    if overtime_hours >= 20:
        burn += random.uniform(1.0, 2.5)
    elif overtime_hours >= 10:
        burn += random.uniform(0.3, 1.0)
    if shift in ("Night", "Rotating 12h"):
        burn += random.uniform(0.5, 1.5)
    if weekly_patient_load > 30:
        burn += random.uniform(0.5, 1.5)
    if mgr_support <= 2:
        burn += random.uniform(0.5, 1.0)
    elif mgr_support >= 4:
        burn -= random.uniform(0.3, 0.8)
    if peer_support >= 4:
        burn -= random.uniform(0.2, 0.5)
    if years_exp > 15:
        burn += random.uniform(0.2, 0.6)
    if unit in ("Emergency", "ICU"):
        burn += random.uniform(0.3, 0.8)
    elif unit in ("Administration", "Facilities"):
        burn -= random.uniform(0.5, 1.0)
    burn += random.uniform(-0.8, 0.8)
    burn = round(max(1.0, min(10.0, burn)), 1)

    return {
        "EmpID": eid,
        "Age": age,
        "Gender": random.choice(["Male", "Female", "Non-Binary"]),
        "Unit": unit,
        "Role": role,
        "CertificationLevel": cert_level,
        "YearsExperience": years_exp,
        "ShiftPattern": shift,
        "WeeklyPatientLoad": weekly_patient_load,
        "OvertimeHoursPerWeek": overtime_hours,
        "AnnualSalary": annual_salary,
        "HasDependents": has_dependents,
        "CommuteMinutes": commute,
        "ManagerSupportScore": mgr_support,
        "PeerSupportScore": peer_support,
        "BurnoutIndex": burn,
    }


def _pinnacle_feedback(employees: list[dict], profile: dict) -> list[dict]:
    """Pinnacle JSON schema: entry_id, staff_number, channel, timestamp,
    prompt, comment, unit"""
    themes = (profile["feedback_positive"], profile["feedback_neutral"], profile["feedback_negative"])
    prompts_map = {
        "wellbeing_check": [
            "How would you rate your current stress level?",
            "Do you feel you have adequate support to manage your workload?",
            "Are you getting enough rest between shifts?",
        ],
        "safety_report": [
            "Have you witnessed any safety concerns related to staffing?",
            "Do you feel comfortable reporting safety issues?",
        ],
        "annual_engagement": [
            "What keeps you working at Pinnacle Healthcare?",
            "What is the single biggest improvement we could make?",
            "How would you describe morale in your unit?",
        ],
        "anonymous_hotline": [
            "Please share any concern or feedback anonymously.",
        ],
    }
    base_date = datetime(2025, 1, 1)
    fb, fid = [], 1

    high_burn = [e for e in employees if e["BurnoutIndex"] >= 7.0]
    low_burn = [e for e in employees if e["BurnoutIndex"] <= 3.5]
    mid_burn = [e for e in employees if 3.5 < e["BurnoutIndex"] < 7.0]

    for emp in random.sample(high_burn, min(len(high_burn), 250)):
        ch = random.choice(["wellbeing_check", "safety_report", "annual_engagement"])
        text = random.choice(themes[2]) if random.random() < 0.80 else random.choice(themes[1])
        fb.append({
            "entry_id": fid, "staff_number": emp["EmpID"],
            "channel": ch,
            "timestamp": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "prompt": random.choice(prompts_map[ch]),
            "comment": text,
            "unit": emp["Unit"],
        })
        fid += 1

    for emp in random.sample(low_burn, min(len(low_burn), 180)):
        ch = random.choice(["wellbeing_check", "annual_engagement"])
        text = random.choice(themes[0]) if random.random() < 0.70 else random.choice(themes[1])
        fb.append({
            "entry_id": fid, "staff_number": emp["EmpID"],
            "channel": ch,
            "timestamp": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "prompt": random.choice(prompts_map[ch]),
            "comment": text,
            "unit": emp["Unit"],
        })
        fid += 1

    for emp in random.sample(mid_burn, min(len(mid_burn), 200)):
        ch = "annual_engagement"
        r = random.random()
        text = random.choice(themes[0]) if r < 0.25 else random.choice(themes[1]) if r < 0.50 else random.choice(themes[2])
        fb.append({
            "entry_id": fid, "staff_number": emp["EmpID"],
            "channel": ch,
            "timestamp": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "prompt": random.choice(prompts_map[ch]),
            "comment": text,
            "unit": emp["Unit"],
        })
        fid += 1

    for _ in range(90):
        emp = random.choice(employees)
        text = random.choice(themes[0] + themes[1] + themes[2])
        fb.append({
            "entry_id": fid, "staff_number": emp["EmpID"],
            "channel": "anonymous_hotline",
            "timestamp": (base_date + timedelta(days=random.randint(0, 365))).isoformat(),
            "prompt": prompts_map["anonymous_hotline"][0],
            "comment": text,
            "unit": emp["Unit"],
        })
        fid += 1
    return fb


# ═════════════════════════════════════════════════════════════════════
#  Unified generation + output
# ═════════════════════════════════════════════════════════════════════

GENERATORS = {
    "novatech":  (NOVATECH,  _novatech_employee,  _novatech_feedback),
    "meridian":  (MERIDIAN,  _meridian_employee,   _meridian_feedback),
    "pinnacle":  (PINNACLE,  _pinnacle_employee,   _pinnacle_feedback),
}


def save_company_data(slug: str):
    profile, emp_fn, fb_fn = GENERATORS[slug]
    company_dir = OUTPUT_DIR / slug
    company_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating data for: {profile['name']}")
    print(f"{'='*60}")

    employees = [emp_fn(eid, profile) for eid in range(1, profile["employee_count"] + 1)]

    # Write CSV
    csv_path = company_dir / "employees.csv"
    fieldnames = list(employees[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(employees)
    print(f"  ✓ employees.csv: {len(employees)} rows, columns: {len(fieldnames)}")
    print(f"    Columns: {', '.join(fieldnames)}")

    # Write JSON
    feedback = fb_fn(employees, profile)
    json_path = company_dir / "feedback.json"
    with open(json_path, "w") as f:
        json.dump(feedback, f, indent=2)
    print(f"  ✓ feedback.json: {len(feedback)} entries")
    fb_keys = list(feedback[0].keys()) if feedback else []
    print(f"    Keys: {', '.join(fb_keys)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic HCM data with distinct schemas per company"
    )
    parser.add_argument(
        "--company",
        choices=list(GENERATORS.keys()),
        help="Generate data for a single company (default: all)",
    )
    args = parser.parse_args()

    if args.company:
        save_company_data(args.company)
    else:
        for slug in GENERATORS:
            save_company_data(slug)

    print(f"\n✅ Done! Files written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
