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
from dotenv import load_dotenv

# LangChain Groq imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

# ==========================
# Load environment variables
# ==========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

scheduler = BackgroundScheduler()
scheduler.start()
from atexit import register

# ==========================
# Utility Functions
# ==========================
def parse_llm_json(text: str):
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
        return json.loads(json_candidate)
    except Exception:
        return None


def fallback_email(buyer_name, project_name, location, highlights, website_url, seller_phone, seller_email):
    body_text = (
        f"We’re excited to introduce {project_name} at {location}.\n\n"
        f"Highlights:\n- " + "\n- ".join(highlights) + "\n\n"
        f"Visit {website_url} or contact us at {seller_phone}, {seller_email}."
    )
    subject = f"{project_name} — Property Update"
    return {"subject": subject, "body_text": body_text}


# ==========================
# Generate Email Template
# ==========================
def generate_chained_email(stage, project_name, location, highlights,
                           email_style, language, word_limit, website_url,
                           seller_phone, seller_email, memory, is_last_followup=False):
    hl_text = ", ".join(highlights)

    # Define stage-specific instructions
    if stage == "intro":
        stage_instruction = (
            "Write an engaging introductory email to a property buyer. "
            "Introduce the project warmly, mention key highlights, and build initial excitement."
        )
    elif is_last_followup:
        stage_instruction = (
            "Write a final follow-up email emphasizing urgency and scarcity — "
            "mention that this might be the last communication. Encourage a prompt response."
        )
    else:
        stage_instruction = (
            "Write a friendly follow-up email that naturally continues the previous conversation. "
            "Briefly acknowledge earlier emails, summarize what was mentioned before, "
            "and add a few new persuasive or emotional points to renew interest."
        )

    # Include previous emails to provide continuity
    previous_emails = "\n\n".join(
        [msg.content for msg in memory.chat_memory.messages if "email generated" in msg.content.lower()]
    ) or "No previous emails yet."

    prompt_template = """
You are a professional real estate email copywriter.

{stage_instruction}

Context:
Project: "{project_name}" at {location}
Highlights: {hl_text}
Website: {website_url}
Seller contact: {seller_phone}, {seller_email}

Here are the previous email(s) that have already been sent to the buyer:
{previous_emails}

Now write the next email in the sequence. It should feel like a natural continuation of this thread — not a repeat.
Add a slightly different tone or focus, and keep it engaging.

Do NOT include greeting or signature.
Only write the main body text of the email in plain, natural language.

Structure:
- 2–4 short paragraphs
- Use bullet points (• or -) for highlights
- Mention the website, phone, and email naturally
- Keep tone: {email_style}
- Language: {language}
- Target length: around {word_limit} words

Return ONLY valid JSON:
{{
  "subject": "string",
  "body_text": "plain text email body with {{buyer_name}}"
}}
"""

    prompt_vars = {
        "stage_instruction": stage_instruction,
        "project_name": project_name,
        "location": location,
        "hl_text": hl_text,
        "website_url": website_url,
        "seller_phone": seller_phone,
        "seller_email": seller_email,
        "previous_emails": previous_emails,
        "email_style": email_style,
        "language": language,
        "word_limit": word_limit,
        "buyer_name": "{buyer_name}",
    }

    # Increase temperature for more variety on followups
    temperature = 0.7 if stage == "intro" else 0.9

    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm | parser
        response = chain.invoke(prompt_vars)
    except Exception as e:
        print(f"⚠️ LLM failed: {e}. Using fallback email.")
        try:
            raw_prompt = ChatPromptTemplate.from_template(prompt_template).format_prompt(**prompt_vars).to_string()
            raw_output = llm.invoke([HumanMessage(content=raw_prompt)]).content
            response = parse_llm_json(raw_output)
        except Exception:
            response = fallback_email(
                "{buyer_name}", project_name, location, highlights, website_url, seller_phone, seller_email
            )

    # Store this email in memory for continuity
    memory.chat_memory.add_user_message(
        f"{stage.capitalize()} email generated:\nSubject: {response['subject']}\nBody: {response['body_text']}"
    )

    return response


# ==========================
# Email Sender
# ==========================
def send_email(to_email, subject, body_text, attachment_path=None, from_email=None, from_password=None):
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain"))  # plain text email

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
# Schedule Email
# ==========================
def schedule_email(buyer_name, buyer_email, stage, run_time, brochure_path,
                   email_subject, email_body, sender_email, sender_password):
    def job(bn=buyer_name, be=buyer_email, stg=stage, sub=email_subject,
            body=email_body, path=brochure_path, se=sender_email, sp=sender_password):
        send_email(
            be,
            sub,
            body,
            attachment_path=path,
            from_email=se,
            from_password=sp,
        )
        print(f"✅ {stg.capitalize()} email sent to {bn} ({be}) at {datetime.now()}")

    scheduler.add_job(job, "date", run_date=run_time)
    print(f"📅 {stage.capitalize()} email scheduled for {buyer_name} at {run_time}")


# ==========================
# Streamlit UI
# ==========================
st.title("🏡 Real Estate Automated Email Campaign (Groq)")

# Gmail Credentials
with st.expander("🔐 Gmail Credentials", expanded=True):
    if "sender_email" not in st.session_state:
        st.session_state.sender_email = ""
    if "sender_password" not in st.session_state:
        st.session_state.sender_password = ""
    st.session_state.sender_email = st.text_input("Sender Gmail", st.session_state.sender_email)
    st.session_state.sender_password = st.text_input("App Password", type="password", value=st.session_state.sender_password)

# Project Details
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

# Schedule
st.subheader("📅 Schedule Emails")
intro_date = st.date_input("Intro Email Date")
intro_time = st.time_input("Intro Email Time")
intro_datetime = datetime.combine(intro_date, intro_time)

num_followups = st.number_input("Number of Follow-up Emails", min_value=0, max_value=10, value=2, step=1)
followup_datetimes = []
for i in range(num_followups):
    f_date = st.date_input(f"Follow-up #{i+1} Date", key=f"f_date_{i}")
    f_time = st.time_input(f"Follow-up #{i+1} Time", key=f"f_time_{i}")
    followup_datetimes.append(datetime.combine(f_date, f_time))

# Start Campaign
if st.button("🚀 Generate & Preview Emails"):
    if not buyers_file:
        st.error("Please upload buyers list first.")
    else:
        buyers_df = pd.read_csv(buyers_file) if buyers_file.name.endswith(".csv") else pd.read_excel(buyers_file)
        hl_list = [h.strip() for h in highlights.splitlines() if h.strip()]

        brochure_path = None
        if brochure_file:
            brochure_path = "brochure.pdf"
            with open(brochure_path, "wb") as f:
                f.write(brochure_file.getbuffer())

        memory = ConversationBufferMemory(memory_key="previous_email", return_messages=True)

        stages = ["intro"] + [f"followup{i+1}" for i in range(num_followups)]
        email_templates = {}

        for idx, stage in enumerate(stages):
            is_last = idx == len(stages) - 1
            email = generate_chained_email(
                stage,
                project_name,
                location,
                hl_list,
                email_style,
                language,
                word_limit,
                seller_website,
                seller_phone,
                seller_email,
                memory,
                is_last_followup=is_last
            )
            # Add generated email to memory for continuity
            memory.chat_memory.add_user_message(
                f"{stage.capitalize()} email generated:\nSubject: {email['subject']}\nBody: {email['body_text']}"
            )
            email_templates[stage] = email

        st.session_state.email_templates = email_templates
        st.session_state.brochure_path = brochure_path
        st.success("✅ Emails generated! You can now preview and edit them below.")


# Preview & Edit Section
if "email_templates" in st.session_state:
    st.subheader("✏️ Preview & Edit Emails Before Sending")

    for stage, template in st.session_state.email_templates.items():
        with st.expander(f"Edit {stage.capitalize()} Email", expanded=True):
            subject = st.text_input(f"{stage.capitalize()} Subject", template["subject"], key=f"sub_{stage}")
            body_text = st.text_area(f"{stage.capitalize()} Body", template["body_text"], key=f"body_{stage}", height=250)
            st.session_state.email_templates[stage]["subject"] = subject
            st.session_state.email_templates[stage]["body_text"] = body_text

    if st.button("📧 Start Email Campaign"):
        buyers_df = pd.read_csv(buyers_file) if buyers_file.name.endswith(".csv") else pd.read_excel(buyers_file)
        brochure_path = st.session_state.brochure_path

        for _, row in buyers_df.iterrows():
            buyer_name = str(row["name"]).strip()
            buyer_email = str(row["email"]).strip()

            for idx, (stage, template) in enumerate(st.session_state.email_templates.items()):
                subject = template["subject"].replace("{buyer_name}", buyer_name)
                body_text = template["body_text"].replace("{buyer_name}", buyer_name)

                full_email_text = (
                    f"Dear {buyer_name},\n\n"
                    f"{body_text}\n\n"
                    f"Best regards,\n{project_name}\n{seller_phone} | {seller_email}"
                )

                send_time = intro_datetime if stage == "intro" else followup_datetimes[idx - 1]

                schedule_email(
                    buyer_name,
                    buyer_email,
                    stage,
                    send_time,
                    brochure_path,
                    subject,
                    full_email_text,
                    sender_email=st.session_state.sender_email,
                    sender_password=st.session_state.sender_password
                )

        st.success("✅ All emails personalized, edited, and scheduled successfully!")
