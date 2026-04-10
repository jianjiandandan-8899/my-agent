"""
SDR Auto-Reply Webhook Server

Flow:
  1. Your agent sends a cold sales email (from lab3)
  2. CEO/prospect replies to the email
  3. MailerSend Inbound catches the reply, POSTs to this server
  4. SDR agent reads the reply and sends a response to keep the conversation going
"""

import os
import asyncio
from fastapi import FastAPI, Request
from mailersend import MailerSendClient, EmailBuilder
from agents import Agent, Runner
from agents.models.openai_provider import OpenAIProvider
from openai import AsyncOpenAI
from agents import set_default_openai_client, function_tool
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/jane/Desktop/claude-code-projects/agent/agents/my_2_openai/.env", override=True)

# --- OpenRouter setup (same as lab3) ---
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
openrouter_client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY")
)
set_default_openai_client(openrouter_client)

FROM_EMAIL = "test@test-3m5jgro0kpzgdpyo.mlsender.net"

# --- Tool: send reply email ---
def make_send_reply_tool(to_email: str):
    @function_tool
    def send_reply(body: str) -> dict:
        """Send a reply email to the prospect"""
        client = MailerSendClient(api_key=os.environ.get("MAILERSEND_API_KEY"))
        email = (EmailBuilder()
            .from_email(FROM_EMAIL)
            .to(to_email)
            .subject("Re: Sales email")
            .text(body)
            .build())
        client.emails.send(email)
        return {"status": "success"}
    return send_reply

# --- SDR Agent ---
sdr_instructions = """
You are Alice, a sales development representative (SDR) at ComplAI.
ComplAI provides an AI-powered SaaS tool for SOC2 compliance and audit preparation.

You receive a reply from a prospect to a cold sales email.
Your job is to:
1. Read their reply carefully
2. Respond in a friendly, professional way to keep the conversation going
3. Answer any questions they have about ComplAI
4. Try to move towards booking a demo or call
5. Use the send_reply tool to send your response

Keep your reply concise and natural — like a real sales rep, not a robot.
"""

# --- FastAPI app ---
app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "SDR webhook server is running"}

@app.post("/webhook")
async def receive_inbound_email(request: Request):
    """
    MailerSend Inbound Parse sends a POST request here when someone replies.
    Extract the sender and email body, then let the SDR agent respond.
    """
    data = await request.json()
    print("Inbound email received:", data)

    # Extract sender email and reply body from MailerSend payload
    try:
        sender_email = data["from"]["email"]
        # MailerSend sends plain text and/or HTML body
        reply_body = data.get("plain") or data.get("text") or data.get("html", "")
    except (KeyError, TypeError) as e:
        print(f"Error parsing payload: {e}")
        print("Raw payload:", data)
        return {"status": "error", "detail": "Could not parse email payload"}

    print(f"Reply from: {sender_email}")
    print(f"Reply body: {reply_body}")

    # Build the SDR agent with a reply tool targeting this sender
    send_reply = make_send_reply_tool(sender_email)
    sdr_agent = Agent(
        name="SDR Agent",
        instructions=sdr_instructions,
        tools=[send_reply],
        model="openai/gpt-4o-mini"
    )

    # Run the agent asynchronously
    message = f"The prospect replied with:\n\n{reply_body}\n\nRespond to keep the conversation going."
    asyncio.create_task(Runner.run(sdr_agent, message))

    return {"status": "received"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
