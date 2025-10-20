import pandas as pd
import numpy as np

# Simulate your dataframe (like the one in your app)
df = pd.DataFrame({
    "department": ["Emergency", "Cardiology", "Trauma", "Unknown", "Emergency", "Cardiology"],
    "ai_risk": ["High", "Medium", "Low", "Medium", "High", "Low"]
})

alerts_list = []

for dept, group in df.groupby("department"):
    if dept == "Unknown":  # skip placeholder
        continue
    high_pct = (group['ai_risk'] == "High").mean()
    medium_pct = (group['ai_risk'] == "Medium").mean()

    if high_pct > 0.3:
        alerts_list.append({
            "department": dept,
            "rate": round(high_pct * 100, 2),
            "status": "High"
        })
    elif medium_pct > 0.3:
        alerts_list.append({
            "department": dept,
            "rate": round(medium_pct * 100, 2),
            "status": "Medium"
        })

print(alerts_list)
