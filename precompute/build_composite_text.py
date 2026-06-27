# composite text for all candidates
def build_composite_text(candidate):

    profile = candidate.get("profile", {})

    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
   

    career_text = " ".join(
        f"{job.get('title', '')} at {job.get('company', '')}. "
        f"{job.get('description', '')}"
        for job in candidate.get("career_history", [])
    )

    education_text = " ".join(
        f"{edu.get('degree', '')} "
        f"{edu.get('field_of_study', '')} "
        f"from {edu.get('institution', '')}"
        for edu in candidate.get("education", [])
    )

    skills_text = "\n".join(
    f"{skill.get('name', '')} "
    f"{skill.get('proficiency', '')} "
    f"{skill.get('duration_months', '')} months"
    for skill in candidate.get("skills", [])
)
    certifications_text = "\n".join(
    cert.get("name", "")
    for cert in candidate.get("certifications", [])
)
    return f"""
Headline: {headline}
Current Title: {current_title}
Current Company: {current_company}
Summary:
{summary}
Career History:
{career_text}
Education:
{education_text}
Skills:
{skills_text}
Certifications:
{certifications_text}
"""
