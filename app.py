import streamlit as st
import pandas as pd
import json
import re
import os
import smtplib
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from groq import Groq
from dotenv import load_dotenv

# ==========================
# Load environment variables
# ==========================
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

scheduler = BackgroundScheduler()
scheduler.start()

# ==========================
# Utility Functions
# ==========================
def parse_llm_json(text: str):
    """Robustly extract and sanitize JSON from LLM output."""
    import json, re

    clean_text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("\r", "")
        .replace("\n", " ")
    )

    match = re.search(r"\{.*\}", clean_text)
    if not match:
        return None

    json_candidate = match.group(0)
    json_candidate = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_candidate)

    try:
        parsed = json.loads(json_candidate)
        return parsed
    except Exception:
        return None


def fallback_email(buyer_name, project_name, location, highlights, website_url, seller_phone, seller_email):
    hl_items = "".join(f"<li>{h}</li>" for h in highlights)
    subject = f"{project_name} — Property Update"
    body_html = f"""
    <p>Dear {buyer_name},</p>
    <p>Here are some highlights of <strong>{project_name}</strong> at {location}:</p>
    <ul>{hl_items}</ul>
    <p>Please visit our website or contact us to learn more.</p>
    <p>Best regards,<br>Sales Team<br>{seller_phone} | {seller_email}</p>
    """
    return {"subject": subject, "body_html": body_html}


# ==========================
# LLM Email Generator
# ==========================
def generate_email_with_llm(buyer_name, project_name, location, highlights, email_style, language, word_limit, website_url, seller_phone, seller_email, stage="intro"):
    client = Groq(api_key=GROQ_API_KEY)
    hl_text = ", ".join(highlights)

    stage_instruction = {
        "intro": "Write a warm and professional introductory email that excites the buyer about the project.",
        "followup1": "Write a friendly follow-up email reminding the buyer about the project details they received earlier.",
        "followup2": "Write a persuasive final reminder emphasizing limited availability and urgency."
    }.get(stage, "Write a professional real estate email.")

    prompt = f"""
You are a professional real estate copywriter.
Task: {stage_instruction}

Project: "{project_name}" located at {location}.
Buyer: {buyer_name}.
Highlights: {hl_text}.
Include greeting, story, highlights, and call-to-action.

IMPORTANT: Always end the email with:
<p>Best regards,<br>Sales Team<br>{seller_phone} | {seller_email}</p>

Mention brochure attachment and website: {website_url}.

Tone: {email_style}, Language: {language}, Length: ~{word_limit} words.

Return ONLY valid JSON in this structure:
{{
  "subject": "string",
  "body_html": "<p>HTML formatted email content here</p>"
}}
Use HTML tags like <p>, <ul>, <li>, <strong>. No markdown or explanations.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        raw = response.choices[0].message.content.strip()
        parsed = parse_llm_json(raw)
        if parsed and "subject" in parsed and "body_html" in parsed:
            return parsed
        else:
            return fallback_email(buyer_name, project_name, location, highlights, website_url, seller_phone, seller_email)
    except Exception:
        return fallback_email(buyer_name, project_name, location, highlights, website_url, seller_phone, seller_email)


# ==========================
# Email Sender
# ==========================
def send_email(to_email, subject, body_html, attachment_path=None, from_email=None, from_password=None):
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_html, "html"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEApplication(f.read(), _subtype="pdf")
                part.add_header("Content-Disposition", "attachment", filename="Brochure.pdf")
                msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(from_email, from_password)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email} ({subject})")
        return True
    except Exception as e:
        print(f"❌ Email failed to {to_email}: {e}")
        return False


# ==========================
# Streamlit UI
# ==========================
st.title("🏡 Real Estate Automated Email Campaign")

with st.expander("🔐 Gmail Credentials", expanded=True):
    sender_email = st.text_input("Sender Gmail")
    sender_password = st.text_input("App Password", type="password")

st.subheader("🏢 Project Details")
col1, col2 = st.columns(2)
with col1:
    project_name = st.text_input("Project Name", "Edenwood Heights")
    location = st.text_input("Location", "Gachibowli, Hyderabad")
with col2:
    seller_phone = st.text_input("Seller Phone", "+91 98765 43210")
    seller_email = st.text_input("Seller Email", "sales@edenwoodheights.com")

seller_website = st.text_input("Website URL", "https://www.edenwoodheights.com")

buyers_file = st.file_uploader("Upload Buyers List (CSV/Excel with 'name' & 'email')", type=["csv", "xlsx"])
brochure_file = st.file_uploader("Upload Brochure (PDF)", type=["pdf"])

highlights = st.text_area("Project Highlights (one per line)", "🌳 Landscaped gardens\n🏊 Swimming pool\n🛣 Excellent connectivity")

email_style = st.selectbox("Email Style", ["Professional", "Friendly", "Formal"], index=0)
language = st.selectbox("Language", ["English", "Hindi", "Telugu"], index=0)
word_limit = st.slider("Word Limit", 80, 500, 250, 10)

# ==========================
# Schedule Section
# ==========================
st.subheader("📅 Schedule Emails")

st.markdown("### Intro Email")
intro_date = st.date_input("Intro Email Date")
intro_time = st.time_input("Intro Email Time")
intro_datetime = datetime.combine(intro_date, intro_time)

st.markdown("### Follow-up Emails")
followup1_date = st.date_input("Follow-up #1 Date")
followup1_time = st.time_input("Follow-up #1 Time")
followup2_date = st.date_input("Follow-up #2 Date")
followup2_time = st.time_input("Follow-up #2 Time")

followup1_datetime = datetime.combine(followup1_date, followup1_time)
followup2_datetime = datetime.combine(followup2_date, followup2_time)

# ==========================
# Scheduling Logic
# ==========================
def schedule_email(buyer_name, buyer_email, stage, run_time, brochure_path, hl_list):
    email_data = generate_email_with_llm(
        buyer_name, project_name, location, hl_list,
        email_style, language, word_limit, seller_website,
        seller_phone, seller_email, stage=stage
    )

    def job():
        send_email(
            buyer_email,
            email_data["subject"],
            email_data["body_html"],
            attachment_path=brochure_path,
            from_email=sender_email,
            from_password=sender_password,
        )
        print(f"✅ {stage.capitalize()} email sent to {buyer_name} ({buyer_email}) at {datetime.now()}")

    scheduler.add_job(job, "date", run_date=run_time)
    print(f"📅 {stage.capitalize()} email scheduled for {buyer_name} at {run_time}")


# ==========================
# Start Campaign
# ==========================
if st.button("🚀 Start Email Campaign"):
    if not buyers_file:
        st.error("Please upload buyers list first.")
    elif not sender_email or not sender_password:
        st.error("Please provide Gmail credentials.")
    else:
        buyers_df = pd.read_csv(buyers_file) if buyers_file.name.endswith(".csv") else pd.read_excel(buyers_file)
        hl_list = [h.strip() for h in highlights.splitlines() if h.strip()]
        brochure_path = None
        if brochure_file:
            brochure_path = "brochure.pdf"
            with open(brochure_path, "wb") as f:
                f.write(brochure_file.getbuffer())

        for _, row in buyers_df.iterrows():
            buyer_name = str(row["name"])
            buyer_email = str(row["email"])

            schedule_email(buyer_name, buyer_email, "intro", intro_datetime, brochure_path, hl_list)
            schedule_email(buyer_name, buyer_email, "followup1", followup1_datetime, brochure_path, hl_list)
            schedule_email(buyer_name, buyer_email, "followup2", followup2_datetime, brochure_path, hl_list)

        st.success("✅ All emails scheduled successfully! They will be sent automatically at the specified times.")
