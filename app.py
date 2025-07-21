import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import pdfplumber
import re
from datetime import datetime
import google.generativeai as genai
import os
from calendar import month_name  # ✅ Fixed missing import

# Set up Google GenAI API
genai.configure(api_key="AIzaSyAbKDqp7HafNGdWAMorvchUA65-dq1-_Z4")
model = genai.GenerativeModel("gemini-2.5-flash")

# Function to extract text from uploaded PDF
def extract_text_from_pdf(file):
    full_text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
    return full_text

# Extract UPI transactions from text
def extract_upi_transactions(text):
    pattern = r'(\d{2}-\d{2}-\d{4}).*?(UPI.*?)\s+INR\s*([\d,]+\.\d{2}).*?(CR|DR)'
    matches = re.findall(pattern, text, re.DOTALL)
    extracted = []
    for date_str, remark, amount, txn_type in matches:
        try:
            date = datetime.strptime(date_str, "%d-%m-%Y").date()
            amount = float(amount.replace(',', ''))
            payee_match = re.search(r'(?:to|from)\s+([\w\.-]+@[\w\.-]+)', remark, re.IGNORECASE)
            payee = payee_match.group(1) if payee_match else "Unknown"
            extracted.append({
                "Date": date,
                "Amount": amount,
                "Type": "Credit" if txn_type == "CR" else "Debit",
                "Remark": remark,
                "Payee": payee
            })
        except:
            continue
    return pd.DataFrame(extracted)

# Spending summary analysis
def analyze_spending(df):
    df['Month'] = df['Date'].apply(lambda x: x.strftime('%B'))
    monthly_spending = df[df['Type'] == 'Debit'].groupby('Month')['Amount'].sum()
    top_payees = df['Payee'].value_counts().head(5)
    total_spent = df[df['Type'] == 'Debit']['Amount'].sum()
    total_received = df[df['Type'] == 'Credit']['Amount'].sum()
    return {
        "total_spent": total_spent,
        "total_received": total_received,
        "monthly_spending": monthly_spending.to_dict(),
        "top_payees": top_payees.to_dict()
    }

# Function to generate advice using LearnLM
def generate_llm_insights(summary):
    prompt = f"""
UPI financial activity:
- Total spent: {summary['total_spent']:.2f}  
- Total received: {summary['total_received']:.2f}  
- Savings: {summary['total_received'] - summary['total_spent']:.2f} ({(summary['total_received'] - summary['total_spent']) / summary['total_received'] * 100:.1f}% of income saved)  
- Monthly spending breakdown: {summary['monthly_spending']}  
- Top payees: {', '.join(summary['top_payees'].keys()) if summary.get('top_payees') else 'N/A'}
Please analyze the UPI spending and income details. Identify key spending habits, comment on savings potential, and suggest actionable budgeting improvements. Output the advice in a clear, bullet-point format.

"""
    response = model.generate_content(prompt)
    return response.text

# Streamlit App
st.title("📊 UPI Statement Analyzer with LearnLM")
uploaded_file = st.file_uploader("📤 Upload your UPI PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    df = extract_upi_transactions(raw_text)

    if df.empty:
        st.warning("⚠️ No UPI transactions found in the uploaded file.")
        st.stop()

    df['Month'] = df['Date'].apply(lambda x: x.strftime('%B'))

    st.subheader("🧾 Extracted Transactions")
    st.dataframe(df)

    st.download_button("📥 Download CSV", df.to_csv(index=False), "upi_transactions.csv")

    # Analysis
    analysis = analyze_spending(df)
    st.subheader("📊 Spending Summary")
    st.json(analysis)

    with st.spinner("💡 Generating personalized financial advice..."):
        insights = generate_llm_insights(analysis)

    st.subheader("🧠 Smart Financial Advice")
    st.markdown(insights)

    # Bar Chart
    monthly_summary = df.groupby(['Month', 'Type'])['Amount'].sum().unstack().fillna(0)
    monthly_summary['Balance'] = monthly_summary.get('Credit', 0) - monthly_summary.get('Debit', 0)
    monthly_summary = monthly_summary.reset_index()

    month_order = list(month_name)[1:]  # Jan to Dec
    monthly_summary['Month'] = pd.Categorical(monthly_summary['Month'], categories=month_order, ordered=True)
    monthly_summary = monthly_summary.sort_values('Month')

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))  # Compact chart
    x = range(len(monthly_summary['Month']))
    width = 0.3

    ax.bar([i - width / 2 for i in x], monthly_summary['Debit'], width, label="Debit", color='#FF6F61')
    ax.bar([i + width / 2 for i in x], monthly_summary['Credit'], width, label="Credit", color='#6B8E23')

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_summary['Month'], rotation=45, ha='right')
    ax.set_ylabel("Amount (INR)", fontsize=10)
    ax.set_title("Monthly UPI Credit vs Debit", fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
