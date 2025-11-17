import streamlit as st
import google.generativeai as genai

GEN_API_KEY = "AIzaSyD3B-M8LrF306rJhPrLuhUVDkx6NUDwTjI"  # Replace with your key!
genai.configure(api_key=GEN_API_KEY)

PROJECT_DESCRIPTION = """
Project Title: Predictive Transaction Intelligence using for BFSI

Project Statement:
This project focuses on developing an AI-driven system that utilizes Large Language Models (LLMs) to analyse historical customer transaction patterns and behavioral data in order to predict future transactions and assess fraud risks in real time. By identifying anomalies and learning fraud indicators from previous data, this solution will help financial institutions enhance transaction security, improve fraud detection, and optimize risk management—without disrupting customer experience.

Outcomes:
Predictive Modeling: Anticipate customer transactions using behavioral and historical data.
Real-Time Risk Assessment: Evaluate transaction legitimacy instantly based on learned patterns.
Fraud Detection: Improve identification of anomalous or high-risk transactions through pattern learning.
Enhanced Financial Security: Strengthen anti-fraud systems, reducing risk exposure for financial institutions.

Modules to be Implemented:
Module 1: Data Collection and Preprocessing
Collect and clean historical transaction data (timestamps, values, locations, customer behaviors).
Normalize and transform the dataset for compatibility with LLM input requirements.
Tag known fraudulent and legitimate transactions for model training.

Module 2: Predictive Transaction Modeling
Fine-tune LLMs for forecasting transaction behavior.
Train models using customer-specific historical data to predict the next likely actions.
Measure prediction accuracy using precision, recall, and F1 scores.

Module 3: Real-Time Fraud Detection Engine
Implement risk detection logic based on model outputs.
Match predicted transactions against known fraud signatures and behavioral deviations.
Generate alerts for high-risk activity in real-time.

Module 4: Deployment and Integration Layer
Deploy predictive models into a live environment.
Integrate the fraud detection engine with existing monitoring systems.
Conduct functional testing for performance, accuracy, and reliability.
"""

def query_gemini(message):
    # RESPOND TO BASIC GREETINGS AND GENERIC QUESTIONS FIRST
    msg_lower = message.strip().lower()
    if msg_lower in ["hi", "hello", "hey"]:
        return "Hello! I am your BFSI Banking AI assistant. Ask me about fraud, transactions, or risk analytics."
    if msg_lower in ["what can you do?", "what can you do", "help", "?", "how can you help?"]:
        return (
            "I can:\n"
            "- Predict the risk and legitimacy of banking transactions\n"
            "- Explain fraud decisions based on AI models\n"
            "- Answer banking/finance project questions\n"
            "- Give guidance on predictive analytics, risk scores, and project modules\n"
        )

    # Otherwise, continue with original Gemini prompt/classifier
    prompt = (
        "You are an expert AI assistant for the following banking/finance project:\n"
        + PROJECT_DESCRIPTION +
        "\nUser message: " + message +
        "\n\n" +
        "Classify the user's transaction as either 'FRAUD' or 'NOT FRAUD', then explain your decision in 1-2 sentences. " +
        "If the information is incomplete, use the most plausible answer based on BFSI domain heuristics. Always respond in this format:\n" +
        "Label: [FRAUD or NOT FRAUD]\nExplanation: [Your reasoning here]"
    )
    try:
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"Gemini API error: {e}"


def chatbot_ui():
    st.title("💬Banking Chatbot( LLM)")
    st.caption("All queries answered and recorded.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", placeholder="Type project-related, fraud, churn, or risk questions...")

    if st.button("Send") and user_input:
        response = query_gemini(user_input)
        st.session_state.chat_history.append(("🧑 You", user_input))
        st.session_state.chat_history.append(("🤖 Bot", response))
        st.rerun()

    for sender, msg in st.session_state.chat_history:
        if sender == "🧑 You":
            st.markdown(f"**{sender}:** {msg}")
        else:
            st.markdown(
                f"<div style='background:#f2f2f2;padding:8px;border-radius:10px'><b>{sender}:</b> {msg}</div>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    chatbot_ui()
