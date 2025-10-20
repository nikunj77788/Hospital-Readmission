# gradio_dashboard.py
import gradio as gr
import pandas as pd
import pickle
from datetime import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import plotly.express as px
import plotly.io as pio

# --- Load ML model ---
with open('readmission_model.pkl', 'rb') as f:
    model = pickle.load(f)

# --- Load dataset ---
df = pd.read_csv('cleaned_diabetes.csv')
TARGET_COL = 'readmitted'

# --- Helper Functions ---
def predict_risk(input_dict):
    X_df = pd.DataFrame([input_dict])
    try:
        pred = model.predict(X_df)[0]
    except:
        X_dummies = pd.get_dummies(X_df)
        if hasattr(model, 'feature_names_in_'):
            for c in model.feature_names_in_:
                if c not in X_dummies.columns:
                    X_dummies[c] = 0
            X_df_aligned = X_dummies[list(model.feature_names_in_)]
            pred = model.predict(X_df_aligned)[0]
    if str(pred).upper() in ['NO','0','N']:
        return "Low"
    elif str(pred)=='>30':
        return "Medium"
    else:
        return "High"

def generate_pdf(name, age, diagnosis, history, treatment):
    input_data = {"name": name, "age": age, "diagnosis": diagnosis, "history": history, "treatment": treatment}
    risk = predict_risk(input_data)
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Patient Readmission Care Plan")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Name: {name}")
    c.drawString(50, 700, f"Predicted Risk: {risk}")
    c.drawString(50, 680, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if risk == "High":
        c.drawString(60, 650, "- Schedule follow-up within 3 days")
        c.drawString(60, 630, "- Ensure medication adherence")
        c.drawString(60, 610, "- Contact care coordinator immediately if symptoms worsen")
    elif risk == "Medium":
        c.drawString(60, 650, "- Schedule follow-up within 7 days")
        c.drawString(60, 630, "- Monitor symptoms and report changes")
        c.drawString(60, 610, "- Review medication with care team")
    else:
        c.drawString(60, 650, "- Standard discharge instructions")
        c.drawString(60, 630, "- Keep regular primary care follow-ups")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def admin_summary():
    dept_col = 'department' if 'department' in df.columns else 'admission_type_id' if 'admission_type_id' in df.columns else '_all_'
    if dept_col == '_all_':
        df['_all_'] = 'All'
        dept_col = '_all_'
    counts = df.groupby(dept_col)[TARGET_COL].value_counts().unstack(fill_value=0).reset_index()
    for col in ['NO', '>30', '<30']:
        if col not in counts.columns:
            counts[col] = 0
    counts['total'] = counts[['NO','>30','<30']].sum(axis=1)
    counts['readmission_rate'] = ((counts['>30'] + counts['<30']) / counts['total']) * 100
    threshold = 20
    counts['alert'] = counts['readmission_rate'].apply(lambda r: "⚠️ High Readmission" if r>threshold else "✅ OK")
    counts['recommended_action'] = counts['alert'].apply(lambda a: "Review follow-ups" if "High" in a else "Continue monitoring")
    fig = px.bar(counts, x=dept_col, y='readmission_rate', color='readmission_rate',
                 labels={dept_col:'Department', 'readmission_rate':'Readmission Rate (%)'},
                 title='Department-wise Readmission Rate (%)')
    fig.update_layout(yaxis_tickformat='%')
    table_data = counts[[dept_col,'readmission_rate','alert','recommended_action']].rename(columns={dept_col:'Department'})
    return fig, table_data

# --- Build Gradio Blocks ---
with gr.Blocks() as demo:
    with gr.Tabs():
        # Doctor Portal
        with gr.Tab("Doctor Portal"):
            gr.Markdown("### Doctor Portal - Predict Patient Readmission Risk")
            name = gr.Textbox(label="Name")
            age = gr.Number(label="Age")
            diagnosis = gr.Textbox(label="Diagnosis")
            history = gr.Textbox(label="Past Visits / History")
            treatment = gr.Textbox(label="Treatment")
            predict_btn = gr.Button("Predict Risk")
            output_risk = gr.Textbox(label="Predicted Risk")
            predict_btn.click(fn=lambda n,a,d,h,t: predict_risk({"name":n,"age":a,"diagnosis":d,"history":h,"treatment":t}),
                              inputs=[name, age, diagnosis, history, treatment],
                              outputs=output_risk)

        # Patient Portal
        with gr.Tab("Patient Portal"):
            gr.Markdown("### Patient Portal - Download Care Plan PDF")
            p_name = gr.Textbox(label="Name")
            p_age = gr.Number(label="Age")
            p_diag = gr.Textbox(label="Diagnosis")
            p_history = gr.Textbox(label="Past Visits / History")
            p_treat = gr.Textbox(label="Treatment")
            pdf_btn = gr.Button("Generate PDF")
            pdf_output = gr.File(label="Download Care Plan PDF")
            pdf_btn.click(fn=generate_pdf,
                          inputs=[p_name, p_age, p_diag, p_history, p_treat],
                          outputs=pdf_output)

        # Admin Dashboard
        with gr.Tab("Admin Dashboard"):
            gr.Markdown("### Admin Dashboard - Department Readmission Overview")
            chart_output = gr.Plot(label="Department Readmission Chart")
            table_output = gr.Dataframe(label="Department Summary")
            refresh_btn = gr.Button("Refresh Data")
            def refresh_admin():
                fig, table = admin_summary()
                return fig, table
            refresh_btn.click(fn=refresh_admin, inputs=[], outputs=[chart_output, table_output])

demo.launch()
