import json
import gzip
import pandas as pd
from datetime import date,datetime

Today = date(2026,6,17)

JD_Skills = ["python","machine learning","deep learning","nlp","llm","embeddings","vector database","rag",
             "retrieval","ranking","recommendation systems","fine-tuning","pytorch","faiss","a/b testing","transformers",
             "LoRA","QLoRA","NDCG","MRR","IR","evaluation"]

Neg_comp = ["tcs","infosys","wipro","hcl","accenture","capgemini",
            "techn mahindra","mindtree","mphasis"]

CAPABILITY_TERMS = {

    "ml": [
        "machine learning",
        "deep learning",
        "supervised learning",
        "unsupervised learning",
        "feature engineering",
        "model training",
        "inference"
    ],

    "retrieval": [
        "retrieval",
        "semantic search",
        "vector search",
        "vector database",
        "embedding",
        "embeddings",
        "faiss",
        "pinecone",
        "weaviate",
        "milvus",
        "bm25",
        "hybrid search",
        "rag",
        "dense retrieval"
    ],

    "ranking": [
        "ranking",
        "learning to rank",
        "ltr",
        "reranking",
        "ranknet",
        "lambdamart",
        "pairwise ranking",
        "listwise ranking"
    ],

    "recommendation": [
        "recommendation",
        "recommender",
        "recommendation engine",
        "candidate generation",
        "two tower",
        "matrix factorization",
        "collaborative filtering"
    ],

    "llm": [
        "llm",
        "large language model",
        "gpt",
        "bert",
        "transformer",
        "lora",
        "qlora",
        "peft",
        "instruction tuning",
        "fine tuning",
        "prompt engineering"
    ],

    "evaluation": [
        "mrr",
        "ndcg",
        "precision",
        "recall",
        "map",
        "auc",
        "roc",
        "f1",
        "a/b testing"
    ]
}

def parse_date(d):
    if d is None:
        return None
    try:
        return datetime.strptime(d[:10],"%Y-%m-%d").date()
    except Exception as e:
        print(f"Error parsing date {d}: {e}")
        return None
    
def days_since(d):
    if d is None:
        return None
    try:
        return (Today - d).days
    except Exception as e:
        print(f"Error calculating days since {d}: {e}")
        return None
    
def profile(c,row):
    p = c["profile"]
    yoe = p["years_of_experience"]
    row["years_of_experience"] = yoe
    if 5<=yoe<=9:
        row["exp_fit"]=1.0
    elif 4<=yoe<5 or 9<yoe<=12:
        row["exp_fit"]=0.7
    elif 3<=yoe<4 or 12<yoe<=15:
        row["exp_fit"]=0.4
    else:
        row["exp_fit"]=0.1
    #dont know what to do with company size
    icity = ["pune","noida","hyderabad","delhi","mumbai","bangalore","bengaluru","bombay","gurugram","gurgaon","chennai"]
    location = p.get("location","").lower()
    if location in icity:
        row["location_fit"] = 1.0
    else:
        row["location_fit"] = 0.1
    row["is_india"]=int(p.get("country","")=="India")

def career_feat(c,row):
    ch = c.get("career_history",[])
    if not ch:
        row["num_jobs"]=0
        row["candid_months_total"]=0
        row["career_gap"]=9999
        row["in_neg_comp"]=0
        row["has_prod_exp"]=0
        row["avg_tenure"]=0
        row["max_tenure"]=0
        row["sus_tenure"]=0
        return
    
    row["num_jobs"]=len(ch)
    tm = sum(h.get("duration_months",0)for h in ch)
    row["candid_months_total"]=tm #honeypot!
    said_months = c["profile"]["years_of_experience"]*12
    row["career_gap"]=abs(said_months-tm)
    current = next((h for h in ch if h.get("is_current")),ch[0])
    ccomp = current.get("company","").strip().lower()
    row["in_neg_comp"]=int(ccomp in Neg_comp)
    row["has_prod_exp"]=int(any(h.get("company","").strip().lower() not in Neg_comp for h in ch))
    tenure = [h.get("duration_months",0) for h in ch]
    row["avg_tenure"]=tm/len(ch) if len(ch) > 0 else 0
    row["max_tenure"]=max(tenure) if tenure else 0
    row["sus_tenure"]=int(max(tenure)>300) #honeypot!
    #add more keys

def edu_feat(c,row):
    edu = c.get("education",[])
    if not edu:
        row["best_tier"]=0
        row["highest_degree"]=0
        return
    tier_map ={
        "tier_1":1.0,
        "tier_2":0.75,
        "tier_3":0.5,
        "tier_4":0.25,
        "tier_5":0.1,
        "unknown":0.3
    }
    row["best_tier"]=max(tier_map.get(e.get("tier","unknown"),0.0) for e in edu)
    row["highest_degree"]=max(e.get("degree",0) for e in edu)
    degree_map = {
        "ph.d": 4, "phd": 4,"PH.D":4,
        "m.tech": 3, "m.s.": 3, "ms": 3, "mtech": 3,
        "mba": 2,"b.tech": 1, "b.e.": 1, "be": 1, "btech": 1, "bachelor": 1}
    best_degree = 0
    for e in edu:
        deg = e.get("degree", "").strip().lower()
        for key, val in degree_map.items():
            if key in deg:
                best_degree = max(best_degree, val)
    row["highest_degree_rank"] = best_degree

def skill_feat(c,row):
    skills = c.get("skills",[])  
    prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    skill_names_lower = {s["name"].lower() for s in skills}
    overlap = skill_names_lower & set(JD_Skills)
    row["jd_skill_overlap_count"] = len(overlap)
    row["jd_skill_overlap_ratio"] = len(overlap) / max(len(JD_Skills), 1)
    matched = [s for s in skills if s["name"].lower() in JD_Skills]
    if matched:
        row["jd_skill_avg"] = sum(
            prof_map.get(s.get("proficiency", "beginner"), 1)
            for s in matched
        ) / len(matched)
        row["jd_skill_avg_months"] = sum(
            s.get("duration_months", 0) for s in matched
        ) / len(matched)
    else:
        row["jd_skill_avg"] = 0.0
        row["jd_skill_avg_months"] = 0.0
    
    row["is_expert"]=sum(
        1 for s in skills if s.get("proficiency")=="expert"
        and s.get("duration_months")<6 #honeypot!
    )
    assessments = c.get("redrob_signals",{}).get("skill_assessment_scores",{})
    row["num_assessments_taken"] = len(assessments)
    row["avg_score"] = (sum(assessments.values()) / len(assessments) / 100
        if assessments else 0.5)
    
def signal_feat(c,row):
    s = c.get("redrob_signals",{})
    last_active = parse_date(s.get("last_active_date"))
    inactive_days = days_since(last_active)
    row["days_since_active"] = inactive_days
    if inactive_days <=14:
        row["active_score"]=1.0
    elif inactive_days <=30:
        row["active_score"]=0.8
    elif inactive_days <=90:
        row["active_score"]=0.4
    elif inactive_days <=120:
        row["active_score"]=0.2
    else:
        row["active_score"]=0.05
    row["open_to_work"]=int(s.get("open_to_work_flag",False))
    notice = s.get("notice_period_days",90)
    row["notice_days"]=notice
    row["notice_score"]=max(0.0,1.0-(notice/120)) #discuss!!!
    row["willing_to_relocate"]=int(s.get("willing_to_relocate",False))
    row["recruiter_response_rate"] = s.get("recruiter_response_rate", 0.5)
    row["interview_completion_rate"] = s.get("interview_completion_rate", 0.5)
    oar = s.get("offer_acceptance_rate",-1)
    row["has_offer_history"] = int(oar > 0)
    row["offer_acceptance_rate"] = oar if oar >= 0 else 0.5
    rt = s.get("avg_response_time_hours",48)
    row["response_time_score"] = max(0.0,1.0 - (rt/240)) #discuss
    verified_email = int(s.get("verified_email",False))
    verified_phone = int(s.get("verified_phone",False))
    linkedin = int(s.get("linkedin_connected",False))
    row["verified_email"] = verified_email
    row["verified_phone"] = verified_phone
    row["linkedin_connected"] = linkedin
    row["trust_score"] = (verified_email + verified_phone + linkedin)/3
    row["profile_completeness"] = s.get("profile_completeness_score", 0)/100
    search_30d = s.get("search_appearance_30d", 0)
    saved_30d = s.get("saved_by_recruiters_30d", 0)
    row["search_appearance_30d"] = search_30d
    row["saved_by_recruiters_30d"] = saved_30d
    row["recruiter_interest_score"] = min(1.0,(min(search_30d, 100) / 100 + min(saved_30d, 10) / 10) / 2)
    gith = s.get("github_activity_score",-1)
    row["has_github"] = int(gith>=0)
    row["github_score"] = max(0,gith)/100

def capability_feat(c, row):
    history = c.get("career_history", [])
    descriptions = " ".join(
        h.get("description", "")
        for h in history
    )
    skills = " ".join( s.get("name", "")
        for s in c.get("skills", [])
    )

    certs = " ".join(cert.get("name", "")
        for cert in c.get("certifications", []))

    text = (descriptions+" "+skills+" "+certs).lower()

    for capability, terms in CAPABILITY_TERMS.items():
        score = 0
        for term in terms:
            if term in text:
                score += 1
        row[f"{capability}_keys"] = score

def build_feature_row(c):
    row = {"candidate_id": c["candidate_id"]}
    profile(c, row)
    career_feat(c, row)
    edu_feat(c, row)
    skill_feat(c, row)
    signal_feat(c, row)
    capability_feat(c, row)
    return row
def load_candids(path): 
    with open(path,"rt") as f: 
        for line in f: 
            if line.strip(): 
                yield json.loads(line)
def load_candids_json(path):
    with open(path) as f:
        return json.load(f)
    
if __name__ == "__main__":
    import os
    import time
    os.makedirs("artifacts", exist_ok=True)
    print("Testing on sample_candidates.json...")
    sample = load_candids_json("data/sample_candidates.json")
    sample_rows = [build_feature_row(c) for c in sample]
    sample_df = pd.DataFrame(sample_rows).set_index("candidate_id")
    print(f"Sample shape: {sample_df.shape}")
    assert len(sample_df) == 50
    assert sample_df.index.is_unique
    print("\nNull counts:")
    print(sample_df.isna().sum())
    print("\nSample validation passed.")
    del sample
    del sample_rows
    del sample_df
    print("\nProcessing full candidates.jsonl...")
    start = time.time()
    rows = []
    for i, c in enumerate(load_candids("./data/candidates.jsonl.gz")):
        rows.append(build_feature_row(c))
        if (i + 1) % 10000 == 0:
            print(f"  {i+1} candidates processed...")
    print(f"Rows collected: {len(rows)}")
    assert len(rows)>90000
    print("Building dataframe...")
    df = pd.DataFrame(rows).set_index("candidate_id")
    print(f"\nFull dataset shape: {df.shape}")
    print(f"\nFull dataset columns: {df.columns.tolist()}")
    df.to_pickle("artifacts/features.pkl")
    print("Saved → artifacts/features.pkl")
    print("Building sample...")
    #Small sample for inspection (to delete later)
    df.sample(
        100,
        random_state=42
    ).to_csv("artifacts/features_sample.csv")
    print("\nSaved:")
    print("artifacts/features.pkl")
    print("artifacts/features_sample.csv")
    elapsed = (time.time()-start)/60
    print(f"\nFinished in {elapsed:.2f} minutes")
    print("\nFeature matrix generation complete!")