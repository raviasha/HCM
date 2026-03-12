"""
Synthetic data generator for HCM AI Insights prototype.

Generates realistic employee attrition data (CSV) and Voice of Employee
feedback (JSON) for 3 distinct company profiles:

  1. NovaTech Solutions  – Tech company, Engineering-heavy, burnout/career stagnation
  2. Meridian Retail Group – Retail, Sales-heavy, compensation/work-life balance
  3. Pinnacle Healthcare   – Healthcare, clinical roles, burnout/management issues

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


# ── Company Profiles ─────────────────────────────────────────────────

COMPANY_PROFILES = {
    "novatech": {
        "name": "NovaTech Solutions",
        "slug": "novatech",
        "employee_count": 1500,
        "attrition_base_rate": 0.14,
        "departments": {
            "Engineering":  {"weight": 0.35, "attrition_boost": 0.10},
            "Product":      {"weight": 0.12, "attrition_boost": 0.02},
            "Sales":        {"weight": 0.15, "attrition_boost": 0.03},
            "Marketing":    {"weight": 0.10, "attrition_boost": 0.00},
            "HR":           {"weight": 0.08, "attrition_boost": -0.02},
            "Finance":      {"weight": 0.08, "attrition_boost": -0.03},
            "Operations":   {"weight": 0.12, "attrition_boost": 0.01},
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
        "attrition_drivers": {
            "OverTime": 1.8,
            "YearsSinceLastPromotion": 1.5,
            "JobSatisfaction_low": 1.4,
            "MonthlyIncome_low": 1.2,
        },
        "overtime_rate": 0.40,
        "feedback_themes": {
            "positive": [
                "I love the technical challenges and innovative projects we work on.",
                "The team culture in engineering is collaborative and supportive.",
                "Great learning opportunities - I've grown so much technically here.",
                "Flexible remote work policy is a huge perk.",
                "Access to cutting-edge tools and technology is excellent.",
                "My manager truly cares about my professional development.",
                "The hackathon events are great for team bonding and innovation.",
                "I appreciate the transparent communication from leadership.",
            ],
            "negative": [
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
            "neutral": [
                "The work is interesting but the pace is intense.",
                "Benefits are standard for the industry, nothing exceptional.",
                "Office space is fine, nothing to complain about.",
                "Communication could be better between teams.",
                "Training budget exists but it's hard to find time to use it.",
            ],
        },
    },
    "meridian": {
        "name": "Meridian Retail Group",
        "slug": "meridian",
        "employee_count": 1800,
        "attrition_base_rate": 0.18,
        "departments": {
            "Sales":        {"weight": 0.30, "attrition_boost": 0.12},
            "Operations":   {"weight": 0.25, "attrition_boost": 0.05},
            "Marketing":    {"weight": 0.10, "attrition_boost": 0.00},
            "HR":           {"weight": 0.07, "attrition_boost": -0.03},
            "Finance":      {"weight": 0.07, "attrition_boost": -0.02},
            "Supply Chain": {"weight": 0.13, "attrition_boost": 0.04},
            "IT":           {"weight": 0.08, "attrition_boost": -0.01},
        },
        "roles": {
            "Sales":        ["Store Associate", "Store Manager", "Regional Manager", "Sales Rep", "Cashier", "Department Lead"],
            "Operations":   ["Warehouse Worker", "Operations Manager", "Logistics Coordinator", "Shift Supervisor", "Inventory Analyst"],
            "Marketing":    ["Marketing Manager", "Visual Merchandiser", "Digital Marketing Specialist", "Brand Analyst"],
            "HR":           ["HR Generalist", "Recruiter", "Training Coordinator", "HR Manager"],
            "Finance":      ["Accountant", "Financial Analyst", "Payroll Specialist", "Controller"],
            "Supply Chain": ["Procurement Analyst", "Supply Chain Manager", "Buyer", "Demand Planner"],
            "IT":           ["IT Support", "Systems Administrator", "Business Analyst", "IT Manager"],
        },
        "attrition_drivers": {
            "MonthlyIncome_low": 2.0,
            "WorkLifeBalance_low": 1.7,
            "OverTime": 1.5,
            "EnvironmentSatisfaction_low": 1.3,
        },
        "overtime_rate": 0.35,
        "feedback_themes": {
            "positive": [
                "I enjoy working with customers and helping them find what they need.",
                "The team on the floor is like family, we look out for each other.",
                "Employee discount is a nice perk.",
                "My store manager is understanding about scheduling needs.",
                "I like the fast-paced environment and no two days are the same.",
                "The company has good values around community involvement.",
            ],
            "negative": [
                "The pay is not enough to live on. I need a second job just to cover rent.",
                "Scheduling is unpredictable - I never know my hours more than a week out.",
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
            "neutral": [
                "The job is straightforward, no surprises.",
                "Some weeks are better than others depending on the schedule.",
                "Training was adequate but could be more thorough.",
                "The break room could use some updating.",
                "Product knowledge sessions are somewhat helpful.",
            ],
        },
    },
    "pinnacle": {
        "name": "Pinnacle Healthcare",
        "slug": "pinnacle",
        "employee_count": 1600,
        "attrition_base_rate": 0.16,
        "departments": {
            "Clinical":     {"weight": 0.35, "attrition_boost": 0.06},
            "Nursing":      {"weight": 0.20, "attrition_boost": 0.08},
            "Administration": {"weight": 0.12, "attrition_boost": -0.02},
            "IT":           {"weight": 0.08, "attrition_boost": -0.01},
            "HR":           {"weight": 0.06, "attrition_boost": -0.03},
            "Finance":      {"weight": 0.07, "attrition_boost": -0.02},
            "Facilities":   {"weight": 0.12, "attrition_boost": 0.03},
        },
        "roles": {
            "Clinical":     ["Physician", "Specialist", "Surgeon", "Resident", "Physician Assistant", "Clinical Researcher"],
            "Nursing":      ["Registered Nurse", "Nurse Practitioner", "Charge Nurse", "Nurse Manager", "CNA"],
            "Administration": ["Admin Assistant", "Office Manager", "Scheduling Coordinator", "Medical Records Clerk"],
            "IT":           ["Systems Administrator", "IT Support", "Health Informatics Analyst", "IT Manager"],
            "HR":           ["HR Generalist", "Recruiter", "Benefits Coordinator", "HR Director"],
            "Finance":      ["Medical Biller", "Financial Analyst", "Revenue Cycle Analyst", "Controller"],
            "Facilities":   ["Maintenance Tech", "Environmental Services", "Facilities Manager", "Safety Officer"],
        },
        "attrition_drivers": {
            "EnvironmentSatisfaction_low": 1.8,
            "OverTime": 1.6,
            "WorkLifeBalance_low": 1.5,
            "JobSatisfaction_low": 1.4,
        },
        "overtime_rate": 0.45,
        "feedback_themes": {
            "positive": [
                "I find deep meaning in helping patients recover and improve their lives.",
                "My colleagues are dedicated professionals who inspire me daily.",
                "The hospital's mission to serve the community keeps me motivated.",
                "Leadership has been investing in new medical equipment which is great.",
                "I appreciate the continuing education reimbursement program.",
                "The patient outcomes we achieve together make the hard days worth it.",
            ],
            "negative": [
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
                "Parking costs, mandatory scrubs we buy ourselves - it all adds up on nurse pay.",
            ],
            "neutral": [
                "The cafeteria food is decent for a hospital.",
                "Orientation was comprehensive but the first week was overwhelming.",
                "IT systems are functional but not intuitive.",
                "The break rooms could use some renovation.",
                "Scheduling is fair most of the time.",
            ],
        },
    },
}

# ── Feedback question prompts ────────────────────────────────────────

QUESTION_PROMPTS = {
    "pulse_survey": [
        "How are you feeling about your work this week?",
        "Is there anything impacting your productivity right now?",
        "How would you rate your energy level at work?",
    ],
    "engagement_survey": [
        "What do you value most about working here?",
        "What is one thing we could improve about your work experience?",
        "How supported do you feel by your direct manager?",
        "Would you recommend this company as a great place to work? Why or why not?",
    ],
    "exit_interview": [
        "What is your primary reason for leaving?",
        "What could we have done differently to retain you?",
        "How would you describe the work culture?",
    ],
    "open_comment": [
        "Share any feedback about your experience at the company.",
        "Is there anything you'd like leadership to know?",
    ],
}


# ── Employee generation ──────────────────────────────────────────────

def _compute_attrition_probability(row: dict, profile: dict) -> float:
    """
    Compute attrition probability for a single employee based on
    the company's specific attrition drivers and the employee's attributes.
    """
    base = profile["attrition_base_rate"]
    dept_info = profile["departments"].get(row["Department"], {})
    base += dept_info.get("attrition_boost", 0)

    drivers = profile["attrition_drivers"]

    # OverTime driver
    if "OverTime" in drivers and row["OverTime"] == "Yes":
        base *= drivers["OverTime"]

    # YearsSinceLastPromotion driver
    if "YearsSinceLastPromotion" in drivers and row["YearsSinceLastPromotion"] >= 4:
        base *= drivers["YearsSinceLastPromotion"]

    # Low satisfaction drivers
    if "JobSatisfaction_low" in drivers and row["JobSatisfaction"] <= 2:
        base *= drivers["JobSatisfaction_low"]

    if "EnvironmentSatisfaction_low" in drivers and row["EnvironmentSatisfaction"] <= 2:
        base *= drivers["EnvironmentSatisfaction_low"]

    if "WorkLifeBalance_low" in drivers and row["WorkLifeBalance"] <= 2:
        base *= drivers["WorkLifeBalance_low"]

    # Low income driver
    if "MonthlyIncome_low" in drivers and row["MonthlyIncome"] < 4000:
        base *= drivers["MonthlyIncome_low"]

    # Age effect (younger = slightly higher attrition)
    if row["Age"] < 30:
        base *= 1.15
    elif row["Age"] > 50:
        base *= 0.85

    return min(base, 0.85)  # cap


def generate_employees(profile: dict) -> list[dict]:
    """Generate synthetic employee records for a company profile."""
    departments = list(profile["departments"].keys())
    dept_weights = [profile["departments"][d]["weight"] for d in departments]

    employees = []
    for eid in range(1, profile["employee_count"] + 1):
        dept = random.choices(departments, weights=dept_weights, k=1)[0]
        roles = profile["roles"][dept]
        job_role = random.choice(roles)

        job_level = random.choices([1, 2, 3, 4, 5], weights=[25, 30, 25, 15, 5], k=1)[0]
        age = random.randint(22, 62)
        years_at_company = random.randint(0, min(age - 22, 30))
        years_in_role = random.randint(0, years_at_company)
        years_since_promo = random.randint(0, years_at_company)
        total_working_years = random.randint(years_at_company, age - 18)

        # Income correlated with job level
        income_base = {1: 3000, 2: 5000, 3: 7500, 4: 10000, 5: 14000}
        monthly_income = int(income_base[job_level] * random.uniform(0.8, 1.4))

        overtime = "Yes" if random.random() < profile["overtime_rate"] else "No"

        row = {
            "EmployeeID": eid,
            "Age": age,
            "Gender": random.choice(["Male", "Female", "Non-Binary"]),
            "MaritalStatus": random.choice(["Single", "Married", "Divorced"]),
            "Department": dept,
            "JobRole": job_role,
            "JobLevel": job_level,
            "MonthlyIncome": monthly_income,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_role,
            "YearsSinceLastPromotion": years_since_promo,
            "TotalWorkingYears": total_working_years,
            "OverTime": overtime,
            "DistanceFromHome": random.randint(1, 50),
            "Education": random.randint(1, 5),
            "PerformanceRating": random.choices([1, 2, 3, 4], weights=[5, 15, 50, 30], k=1)[0],
            "JobSatisfaction": random.randint(1, 4),
            "EnvironmentSatisfaction": random.randint(1, 4),
            "WorkLifeBalance": random.randint(1, 4),
            "RelationshipSatisfaction": random.randint(1, 4),
            "NumCompaniesWorked": random.randint(0, 9),
            "TrainingTimesLastYear": random.randint(0, 6),
            "StockOptionLevel": random.randint(0, 3),
            "PercentSalaryHike": random.randint(11, 25),
            "EngagementScore": round(random.uniform(3.0, 9.5), 1),
        }

        # Determine attrition
        prob = _compute_attrition_probability(row, profile)
        row["Attrition"] = "Yes" if random.random() < prob else "No"

        employees.append(row)

    return employees


# ── Feedback generation ──────────────────────────────────────────────

def generate_feedback(employees: list[dict], profile: dict) -> list[dict]:
    """Generate Voice of Employee feedback entries for a company."""
    themes = profile["feedback_themes"]
    feedback_list = []
    fid = 1

    # Separate attrited employees for exit interviews
    attrited = [e for e in employees if e["Attrition"] == "Yes"]
    active = [e for e in employees if e["Attrition"] == "No"]

    base_date = datetime(2025, 1, 1)

    # Exit interviews for all attrited employees
    for emp in attrited:
        date = base_date + timedelta(days=random.randint(0, 365))
        prompt = random.choice(QUESTION_PROMPTS["exit_interview"])
        # Exit interviews skew heavily negative
        if random.random() < 0.80:
            text = random.choice(themes["negative"])
        elif random.random() < 0.5:
            text = random.choice(themes["neutral"])
        else:
            text = random.choice(themes["positive"])

        feedback_list.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": "exit_interview",
            "date": date.strftime("%Y-%m-%d"),
            "question_prompt": prompt,
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1

    # Pulse surveys and engagement surveys for active employees (sample)
    survey_sample = random.sample(active, min(len(active), 350))
    for emp in survey_sample:
        feedback_type = random.choice(["pulse_survey", "engagement_survey"])
        date = base_date + timedelta(days=random.randint(0, 365))
        prompt = random.choice(QUESTION_PROMPTS[feedback_type])

        # Mixed sentiment for active employees
        r = random.random()
        if r < 0.35:
            text = random.choice(themes["positive"])
        elif r < 0.60:
            text = random.choice(themes["neutral"])
        else:
            text = random.choice(themes["negative"])

        feedback_list.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": feedback_type,
            "date": date.strftime("%Y-%m-%d"),
            "question_prompt": prompt,
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1

    # Open comments (anonymous, any employee)
    for _ in range(80):
        emp = random.choice(employees)
        date = base_date + timedelta(days=random.randint(0, 365))
        prompt = random.choice(QUESTION_PROMPTS["open_comment"])

        r = random.random()
        if r < 0.25:
            text = random.choice(themes["positive"])
        elif r < 0.45:
            text = random.choice(themes["neutral"])
        else:
            text = random.choice(themes["negative"])

        feedback_list.append({
            "feedback_id": fid,
            "employee_id": emp["EmployeeID"],
            "feedback_type": "open_comment",
            "date": date.strftime("%Y-%m-%d"),
            "question_prompt": prompt,
            "response_text": text,
            "department": emp["Department"],
        })
        fid += 1

    return feedback_list


# ── File output ──────────────────────────────────────────────────────

def save_company_data(profile: dict):
    """Generate and save data for a single company profile."""
    slug = profile["slug"]
    company_dir = OUTPUT_DIR / slug
    company_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Generating data for: {profile['name']}")
    print(f"{'='*60}")

    # Generate employee records
    employees = generate_employees(profile)
    attrition_count = sum(1 for e in employees if e["Attrition"] == "Yes")
    attrition_rate = attrition_count / len(employees) * 100

    # Write CSV
    csv_path = company_dir / "employees.csv"
    fieldnames = list(employees[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(employees)

    print(f"  ✓ employees.csv: {len(employees)} rows, "
          f"attrition rate: {attrition_rate:.1f}%")

    # Generate feedback
    feedback = generate_feedback(employees, profile)

    # Write JSON
    json_path = company_dir / "feedback.json"
    with open(json_path, "w") as f:
        json.dump(feedback, f, indent=2)

    print(f"  ✓ feedback.json: {len(feedback)} entries")

    # Print department breakdown
    from collections import Counter
    dept_counter = Counter(e["Department"] for e in employees)
    dept_attrition = Counter(
        e["Department"] for e in employees if e["Attrition"] == "Yes"
    )
    print(f"\n  Department breakdown:")
    for dept in sorted(dept_counter.keys()):
        total = dept_counter[dept]
        att = dept_attrition.get(dept, 0)
        print(f"    {dept:20s}: {total:4d} employees, "
              f"{att:3d} attrited ({att/total*100:.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic HCM data for demo companies"
    )
    parser.add_argument(
        "--company",
        choices=list(COMPANY_PROFILES.keys()) + ["all"],
        default="all",
        help="Which company to generate (default: all)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.company == "all":
        for profile in COMPANY_PROFILES.values():
            save_company_data(profile)
    else:
        save_company_data(COMPANY_PROFILES[args.company])

    print(f"\n✅ Done! Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
