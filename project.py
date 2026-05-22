# project.py


import pandas as pd
import numpy as np
from pathlib import Path

import plotly.express as px


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    keys = ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']
    dct = {key: [] for key in keys}

    for col in grades.columns:
        col_lower = col.lower()
        if '-' in col or 'free_response' in col_lower:
            continue
        if 'checkpoint' in col_lower:
            dct['checkpoint'].append(col)
        else:
            for key in keys:
                if col_lower.startswith(key):
                    dct[key].append(col)
                    break
    for key in dct:
        dct[key].sort()

    return dct


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    names = get_assignment_names(grades)
    project_cols = names.get('project', [])
    if not project_cols:
        return pd.Series(0.0, index=grades.index)

    per_project_props = []

    for base in project_cols:
        earned_auto = grades[base].fillna(0)

        fr_col = f"{base}_free_response"
        earned_fr = grades[fr_col].fillna(0) if fr_col in grades.columns else 0
        earned = earned_auto + earned_fr

        
        max_auto_col = f"{base} - Max Points"
        max_fr_col = f"{fr_col} - Max Points"
        max_auto = grades[max_auto_col]
        max_fr = grades[max_fr_col] if max_fr_col in grades.columns else 0
        max_total = (max_auto + max_fr)

        prop = earned.divide(max_total.replace(0, np.nan)).fillna(0)
        per_project_props.append(prop)

    
    out = pd.concat(per_project_props, axis=1).mean(axis=1)
    out.name = "Projects"
    return out

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def lateness_penalty(col):
    td = pd.to_timedelta(col, errors="coerce").fillna(pd.Timedelta(0))

    td_eff = (td - pd.Timedelta(hours=2)).clip(lower=pd.Timedelta(0))

    one_week = pd.Timedelta(weeks=1)
    two_weeks = pd.Timedelta(weeks=2)

    conditions = [
        td_eff == pd.Timedelta(0),
        (td_eff > pd.Timedelta(0)) & (td_eff <= one_week),
        (td_eff > one_week) & (td_eff <= two_weeks),
    ]
    choices = [1.0, 0.9, 0.7]

    out = pd.Series(np.select(conditions, choices, default=0.4), index=col.index)
    out.name = getattr(col, "name", None)
    return out


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def process_labs(grades):
    names = get_assignment_names(grades)
    lab_cols = names.get('lab', [])
    
    
    if not lab_cols:
        return pd.DataFrame(index=grades.index)

    processed = pd.DataFrame(index=grades.index)

    for lab in lab_cols:
        
        raw = grades[lab].fillna(0)
        max_col = f"{lab} - Max Points"
        max_points = grades[max_col].replace(0, np.nan)
        normalized = raw / max_points
        normalized = normalized.fillna(0)
        late_col = f"{lab} - Lateness (H:M:S)"
        if late_col in grades.columns:
            multiplier = lateness_penalty(grades[late_col])
        else:
            multiplier = 1.0

        adjusted = normalized * multiplier
        processed[lab] = adjusted

    return processed
    


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def lab_total(processed):
    if processed.empty:
        return pd.Series(0.0, index=processed.index)
    if processed.shape[1] == 1:
        return processed.iloc[:, 0].fillna(0)
    
    dropped = processed.apply(lambda row: np.sort(row)[1:] if row.notna().any() else [0], axis=1)
    total = dropped.apply(np.mean)
    
    total = total.clip(0, 1)
    total.name = "Lab Total"
    return total


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def total_points(grades):
    def avg_normalized(cols):
        if not cols:
            return pd.Series(0.0, index=grades.index)
        pieces = []
        for base in cols:
            earned = grades[base].fillna(0)
            max_col = f"{base} - Max Points"
            denom = grades[max_col].replace(0, np.nan)
            pieces.append(earned.divide(denom).fillna(0))
        return pd.concat(pieces, axis=1).mean(axis=1)

    names = get_assignment_names(grades)
    labs_prop = lab_total(process_labs(grades))                
    projects_prop = projects_total(grades)                      
    checkpoints_prop = avg_normalized(names.get('checkpoint', []))
    discussions_prop = avg_normalized(names.get('disc', []))    
    midterm_prop = avg_normalized(names.get('midterm', []))    
    final_prop = avg_normalized(names.get('final', []))

    total = (
        0.20 * labs_prop +
        0.30 * projects_prop +
        0.025 * checkpoints_prop +
        0.025 * discussions_prop +
        0.15 * midterm_prop +
        0.30 * final_prop
    )

    total = total.clip(0, 1)
    total.name = "Total"
    return total


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def final_grades(total):
    bins = [-np.inf, 0.6, 0.7, 0.8, 0.9, np.inf]
    labels = ['F', 'D', 'C', 'B', 'A']
    return pd.cut(total, bins=bins, labels=labels, right=False, include_lowest=True)

def letter_proportions(total):
    letters = final_grades(total)
    props = letters.value_counts(normalize=True)
    return props.sort_values(ascending=False)


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def raw_redemption(final_breakdown, question_numbers):
    qcols = [final_breakdown.columns[i] for i in question_numbers]

    
    earned = final_breakdown[qcols].fillna(0).sum(axis=1)

    
    qmax = final_breakdown[qcols].max(skipna=True)
    denom = qmax.sum()

    
    raw_prop = (earned / denom) if denom > 0 else pd.Series(0.0, index=final_breakdown.index)

    out = pd.DataFrame({
        'PID': final_breakdown['PID'],
        'Raw Redemption Score': raw_prop
    })
    return out
    
def combine_grades(grades, raw_redemption_scores):
    merged = grades.merge(
        raw_redemption_scores[['PID', 'Raw Redemption Score']],
        on='PID',
        how='left'
    )
    merged['Raw Redemption Score'] = merged['Raw Redemption Score'].fillna(0.0)
    return merged


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def z_score(ser):
    mu = ser.mean()
    sd = ser.std(ddof=0)
    return (ser - mu) / sd
    
def add_post_redemption(grades_combined):
    out = grades_combined.copy()

    names = get_assignment_names(out)
    midterms = names.get('midterm', [])

    if not midterms:
        out['Midterm Score Pre-Redemption'] = 0.0
        out['Midterm Score Post-Redemption'] = 0.0
        return out

    parts = []
    for base in midterms:
        earned = out[base]
        denom = out[f"{base} - Max Points"].replace(0, np.nan)
        parts.append(earned.divide(denom))
    mid_prop = pd.concat(parts, axis=1).mean(axis=1)

    
    mid_prop_filled = mid_prop.fillna(0.0).clip(lower=0.0, upper=1.0)

    
    out['Midterm Score Pre-Redemption'] = mid_prop_filled

             
    z_mid = z_score(mid_prop_filled)
    z_red = z_score(out['Raw Redemption Score'])

    
    mid_mu = mid_prop_filled.mean()
    mid_sd = mid_prop_filled.std(ddof=0)
    mid_from_redemption = z_red * mid_sd + mid_mu

    
    improved = z_red > z_mid
    mid_post = mid_prop_filled.where(~improved, mid_from_redemption)
    
    mid_post = np.maximum(mid_post, mid_prop_filled)
    mid_post = mid_post.clip(lower=0.0, upper=1.0)

    out['Midterm Score Post-Redemption'] = mid_post

    return out


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def total_points_post_redemption(grades_combined):
    
    df = grades_combined.copy()  
    
    if ('Midterm Score Pre-Redemption' not in df.columns or
        'Midterm Score Post-Redemption' not in df.columns):
        df = add_post_redemption(df)

    base_total = total_points(df)
    pre_mid = df['Midterm Score Pre-Redemption']
    post_mid = df['Midterm Score Post-Redemption']

    total_post = base_total + 0.15 * (post_mid - pre_mid)
    return total_post.clip(0, 1).astype(float)
        
def proportion_improved(grades_combined):
    df = grades_combined.copy()

    
    if ('Midterm Score Pre-Redemption' not in df.columns or
        'Midterm Score Post-Redemption' not in df.columns):
        df = add_post_redemption(df)

    
    if len(df) == 0:
        return 0.0

    
    total_pre = total_points(df).astype(float).fillna(0.0)
    
    total_post = total_points_post_redemption(df).astype(float).fillna(0.0)

    
    letters_pre = final_grades(total_pre)
    letters_post = final_grades(total_post)

    
    rank = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
    pre_score = letters_pre.astype(object).map(rank).fillna(0)
    post_score = letters_post.astype(object).map(rank).fillna(0)

   
    improved = (post_score > pre_score)
    return float(improved.mean())


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def section_most_improved(grades_analysis):
    df = grades_analysis.copy()

    
    rank = {'F': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4}
    pre = df['Letter Grade Pre-Redemption'].map(rank)
    post = df['Letter Grade Post-Redemption'].map(rank)

    
    df['improved'] = post > pre

    
    section_improve = df.groupby('Section')['improved'].mean()

    
    return section_improve.idxmax()
    
def top_sections(grades_analysis, t, n):
    df = grades_analysis.copy()

    
    names = get_assignment_names(df)
    final_bases = names.get('final', [])
    if not final_bases:
        
        final_bases = [
            c for c in df.columns
            if c.lower().startswith('final')
            and 'max points' not in c.lower()
            and 'lateness' not in c.lower()
            and 'free_response' not in c.lower()
        ]

    
    if not final_bases:
        return np.array([], dtype=object)

    
    pieces = []
    for base in final_bases:
        earned = df[base].fillna(0)
        max_col = f"{base} - Max Points"
        denom = df[max_col].replace(0, np.nan)
        pieces.append(earned.divide(denom).fillna(0))

    final_prop = pd.concat(pieces, axis=1).mean(axis=1)

    
    counts = (
        df.assign(final_prop=final_prop)
          .groupby('Section')['final_prop']
          .apply(lambda s: (s >= t).sum())
    )

    
    return np.sort(counts[counts >= n].index.values)

# ---------------------------------------------------------------------
# QUESTION 12
# ---------------------------------------------------------------------


def rank_by_section(grades_analysis):
    df = grades_analysis.copy()

    
    if 'Total Points Post-Redemption' not in df.columns:
        df['Total Points Post-Redemption'] = total_points_post_redemption(df)

    
    sec_sizes = df.groupby('Section').size()
    n = int(sec_sizes.max()) if len(sec_sizes) else 0
    if n == 0:
        out = pd.DataFrame()
        out.index.name = 'Section Rank'
        out.columns.name = 'Section'
        return out

    
    df_sorted = df.sort_values(['Section', 'Total Points Post-Redemption', 'PID'],
                               ascending=[True, False, True])

    
    ranked = (
        df_sorted
        .groupby('Section', group_keys=True)
        .apply(lambda g: pd.DataFrame({
            'PID': g['PID'].to_numpy(),
            'Section Rank': np.arange(1, len(g) + 1, dtype=int)
        }))
        .reset_index(level=0)
        .rename(columns={'level_0': 'Section'})
    )

    
    table = ranked.pivot(index='Section Rank', columns='Section', values='PID')

    
    def sort_key(s):
        try:
            return int(s[1:]) 
        except Exception:
            return float('inf')

    table = (
        table
        .reindex(index=range(1, n + 1))  
        .reindex(columns=sorted(table.columns, key=sort_key))
        .fillna('')  
    )

    table.index.name = 'Section Rank'
    table.columns.name = 'Section'
    return table







# ---------------------------------------------------------------------
# QUESTION 13
# ---------------------------------------------------------------------


def letter_grade_heat_map(grades_analysis):
    df = grades_analysis.copy()

    
    if 'Letter Grade Post-Redemption' not in df.columns:
        df = df.assign(**{
            'Letter Grade Post-Redemption': final_grades(
                total_points_post_redemption(df)
            )
        })

    
    letters = ['A', 'B', 'C', 'D', 'F']
    cat = pd.api.types.CategoricalDtype(categories=letters, ordered=True)
    df['LetterCat'] = df['Letter Grade Post-Redemption'].astype(cat)

    
    def sec_key(s):
        s = str(s)
        try:
            return int(s[1:])
        except Exception:
            return float('inf')

    sections = sorted(df['Section'].dropna().unique(), key=sec_key)

    
    mat = (
        pd.crosstab(df['LetterCat'], df['Section'], normalize='columns')
          .reindex(index=letters)
          .reindex(columns=sections, fill_value=0.0)
          .fillna(0.0)
    )

    fig = px.imshow(
        mat.values,
        x=mat.columns,
        y=mat.index,
        color_continuous_scale='YlGnBu',
        zmin=0, zmax=1,
        aspect='auto'
    )
    fig.update_layout(
        title='Distribution of Letter Grades by Section',
        xaxis_title='Section',
        yaxis_title='Letter Grade Post-Redemption'
    )
    return fig
