# Course Grades Analysis Toolkit

A Python and pandas project that analyzes course grade data, computes final course scores, applies lateness penalties, models redemption-based grading adjustments, ranks students by section, and visualizes grade distributions across discussion sections.

This project was built as a data analysis toolkit for working with a realistic course gradebook. It focuses on turning raw grade data into meaningful student-level and section-level insights using clean, reusable Python functions.

## Overview

The toolkit processes a course gradebook containing labs, projects, checkpoints, discussion scores, exams, lateness data, and final exam breakdowns. It calculates normalized assignment scores, applies course grading rules, determines letter grades, evaluates post-redemption score changes, and creates summary outputs that help analyze student performance.

The project demonstrates skills in:

- Data cleaning and transformation with pandas
- Numerical computation with NumPy
- Grade calculation logic and weighted scoring
- Handling missing values and late submissions
- Merging multiple datasets
- Section-level ranking and performance analysis
- Data visualization with Plotly
- Writing modular, testable Python functions

## Features

### Assignment Detection

Automatically identifies assignment groups from gradebook columns, including:

- Labs
- Projects
- Project checkpoints
- Discussions
- Midterm exams
- Final exams

This allows the toolkit to work with structured gradebook data without manually hardcoding every assignment name.

### Project Score Calculation

Computes project totals by combining autograded and free-response components, normalizing scores by max points, and averaging across project assignments.

### Lateness Penalties

Applies late submission penalties based on time submitted after the deadline. The system includes a grace period and reduces credit depending on how late the submission is.

### Lab Processing

Normalizes lab scores, applies lateness penalties, drops the lowest lab score, and calculates each student’s final lab total.

### Weighted Course Totals

Calculates final course totals using a weighted grading scheme across labs, projects, checkpoints, discussions, midterms, and finals.

### Letter Grades

Converts numerical course totals into letter grades using defined grade cutoffs.

### Redemption Scoring

Processes final exam question-level data to calculate raw redemption scores. These scores are then used to model post-redemption midterm improvements for students whose final exam performance shows stronger mastery.

### Pre- and Post-Redemption Comparison

Compares course performance before and after redemption adjustments, including whether students improved their final letter grade.

### Section Analysis

Provides tools to:

- Find the section with the highest proportion of improved students
- Identify top-performing sections based on final exam thresholds
- Rank students within each section by post-redemption total points

### Grade Distribution Heat Map

Generates an interactive Plotly heat map showing the distribution of post-redemption letter grades by section.

## Tech Stack

- Python
- pandas
- NumPy
- Plotly Express
- Jupyter Notebook
- Otter Grader

## Project Structure

```text
.
├── project.py
├── project.ipynb
├── project-validation.py
├── data/
│   ├── grades.csv
│   ├── final_exam_breakdown.csv
│   └── heatmap-example.png
└── README.md