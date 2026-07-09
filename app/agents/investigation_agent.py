"""
Investigation report agent.

Turns a SHAP-based fraud explanation into a short, plain-English report
a human fraud analyst could act on. The LLM only explains and summarizes
a decision the ML model already made — it does not re-decide fraud itself.
"""

import json

from groq import Groq

from app.core.config import settings

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a fraud investigation assistant for a bank's fraud operations team.
You are given a transaction that a machine learning model has already flagged, along with the
model's explanation (SHAP feature contributions) for why. Your job is to write a SHORT, plain-English
summary a human fraud analyst can read in 10 seconds to decide whether to act.

Do NOT re-decide whether it's fraud — the model already scored it. Your job is only to explain
the model's reasoning in plain language and suggest a next action.

Respond ONLY with valid JSON in this exact shape, no other text:
{
  "summary": "2-3 sentence plain-English explanation of why this was flagged",
  "risk_level": "low" | "medium" | "high" | "critical",
  "recommended_action": "one short sentence recommending what the analyst should do next"
}
"""


class InvestigationReportAgent:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_report(self, transaction: dict, prediction_result: dict) -> dict:
        user_prompt = f"""
Transaction details:
- Type: {transaction['type']}
- Amount: {transaction['amount']}
- Origin account balance before transaction: {transaction['oldbalance_org']}
- Destination account balance before transaction: {transaction['oldbalance_dest']}

Model output:
- Fraud probability: {prediction_result['fraud_probability']:.4f}
- Flagged: {prediction_result['is_flagged']}

Top contributing features (SHAP values — positive pushes toward fraud, negative pushes away):
{json.dumps(prediction_result['top_contributing_features'], indent=2)}
"""

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content
        return json.loads(raw_content)


investigation_report_agent = InvestigationReportAgent()