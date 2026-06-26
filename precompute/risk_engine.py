import json
import pickle
import pandas as pd
df = pd.read_pickle("artifacts/features.pkl")
def detect_honeypots(df):
    df["risk_score"]=0
    df["risk_level"]="SAFE"
    df["honeypot"]=False
    df["risk_reasons"]=""
    RULE_WEIGHTS = {

    "experience_gap":3,

    "tenure":2,

    "profile_trust":3,

    "activity":2,

    "skill_stuffing":3,

    "response":2,

    "senior_without_skills":2,

    "interview":1,

    "github":1,

    "inactive_product":1
    }
    for idx,row in df.iterrows():
        risk =0
        reasons =[]
        if row["career_gap"]>36:
            risk += RULE_WEIGHTS["experience_gap"]
            reasons.append("Large experience gap")
        if(row["sus_tenure"]==1):
            risk += RULE_WEIGHTS["tenure"]
            reasons.append("Suspiciously long tenure")
        if(row["profile_completeness"]>=0.85 and row["trust_score"]==0):
            risk += RULE_WEIGHTS["profile_trust"]
            reasons.append("High profile completeness but not verified")
        if(row["active_score"]<=0.05 and row["saved_by_recruiters_30d"]>20):
            risk += RULE_WEIGHTS["activity"]
            reasons.append("Inactive profile but saved by many recruiters")
        if(row["jd_skill_overlap_ratio"]>=0.9 and row["avg_score"]<0.3 and row["num_assessments_taken"]==0):
            risk += RULE_WEIGHTS["skill_stuffing"]
            reasons.append("Skill stuffing detected")
        if(row["recruiter_interest_score"]>0.8 and row["recruiter_response_rate"]<0.15):
            risk += RULE_WEIGHTS["response"]
            reasons.append("High recruiter interest but low response rate")
        if(row["years_of_experience"]>5 and row["jd_skill_overlap_count"]==0):
            risk += RULE_WEIGHTS["senior_without_skills"]
            reasons.append("Senior candidate without relevant skills")
        if(row["response_time_score"]>0.9 and row["interview_completion_rate"]<0.2):
            risk += RULE_WEIGHTS["interview"]
            reasons.append("High response time but low interview completion")
        if(row["github_score"]>0.8 and row["ml_keys"]==0 or row["llm_keys"]==0):
            risk += RULE_WEIGHTS["github"]
            reasons.append("High GitHub score but no relevant skills")
        if(row["has_prod_exp"]==1 and row["active_score"]<0.2 and row["recruiter_response_rate"]<0.1):
            risk += RULE_WEIGHTS["inactive_product"]
            reasons.append("Has product experience but inactive and low recruiter response")
        df.at[idx,"risk_score"]=risk
        df.at[idx,"risk_reasons"]="; ".join(reasons)
        if risk>5:
            df.at[idx,"risk_level"]="HONEYPOT"
            df.at[idx,"honeypot"]=True
        elif risk>2:
            df.at[idx,"risk_level"]="REVIEW"
        else:
            df.at[idx,"risk_level"]="SAFE"
    return df

if __name__ == "__main__":
    import json, os
    os.makedirs("artifacts", exist_ok=True)

    print("Loading features...")
    df = detect_honeypots(df)
    print(df.columns.tolist())
    df.to_pickle("artifacts/risk_scores.pkl")
    print("Saved → artifacts/risk_scores.pkl")
    df.to_csv("artifacts/risk_scores.csv", index=True)
    print("Saved → artifacts/risk_scores.csv")
    print(df.columns.tolist())
    honeypot_ids = df.index[df["honeypot"]].tolist()
    with open("artifacts/honeypot_flags.json","w") as f:
        json.dump(honeypot_ids,f,indent=4)
    print(f"Saved → {len(honeypot_ids)} honeypot flags")
    summary = {
    "total_candidates": int(len(df)),
    "safe": int((df["risk_level"] == "SAFE").sum()),
    "review": int((df["risk_level"] == "REVIEW").sum()),
    "honeypot": int((df["risk_level"] == "HONEYPOT").sum()),
    "average_risk_score": float(df["risk_score"].mean()),
    "maximum_risk_score": int(df["risk_score"].max()),
    "minimum_risk_score": int(df["risk_score"].min())}

    with open("artifacts/risk_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved → artifacts/risk_summary.json")
    df.sample(
    100,
    random_state=42).to_csv(
    "artifacts/risk_sample.csv")
    print("Saved → artifacts/risk_sample.csv")
    print(df[df["risk_level"]=="HONEYPOT"].head(10))
    print(df["risk_score"].value_counts().sort_index())