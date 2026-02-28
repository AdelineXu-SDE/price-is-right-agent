import os
from typing import Optional

import requests
from litellm import completion

from agents.deals import Opportunity
from agents.agent import Agent

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.WHITE

    # Optional: used only for copywriting
    MODEL = os.getenv("COPYWRITER_MODEL", "claude-sonnet-4-5")

    def __init__(self):
        """Send push notifications (Pushover) with optional LLM-written copy."""
        self.log("Messaging Agent is initializing")
        self.pushover_user: Optional[str] = os.getenv("PUSHOVER_USER")
        self.pushover_token: Optional[str] = os.getenv("PUSHOVER_TOKEN")
        self.log("Messaging Agent ready")

    def push(self, text: str) -> None:
        """Send a push notification via Pushover, if credentials are configured."""
        if not self.pushover_user or not self.pushover_token:
            self.log("Pushover credentials not set; skipping notification")
            self.log(f"(Preview) {text}")
            return

        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": text,
            "sound": "cashregister",
        }
        try:
            resp = requests.post(PUSHOVER_URL, data=payload, timeout=10)
            resp.raise_for_status()
        except Exception as e:
            self.log(f"Pushover send failed: {e}")

    def craft_message(self, description: str, deal_price: float, estimated_true_value: float) -> str:
        """Use an LLM to craft a short, exciting push message. Falls back to a template if LLM fails."""
        user_prompt = (
            "Write a 2-3 sentence push notification about this deal. Be concise and exciting.\n"
            f"Item: {description}\n"
            f"Offered Price: {deal_price}\n"
            f"Estimated Value: {estimated_true_value}\n"
            "Return ONLY the message."
        )
        try:
            response = completion(model=self.MODEL, messages=[
                                  {"role": "user", "content": user_prompt}])
            return response.choices[0].message.content
        except Exception:
            # Safe fallback: no LLM required
            return (
                f"Deal alert: ${deal_price:.2f} (estimated value ${estimated_true_value:.2f}). "
                "Tap to view details!"
            )

    def notify(self, description: str, deal_price: float, estimated_true_value: float, url: str) -> None:
        """Craft and send a notification about the best deal."""
        self.log("Messaging Agent is crafting message")
        text = self.craft_message(
            description, deal_price, estimated_true_value)
        self.push((text[:200] + "... " + url) if len(text)
                  > 200 else (text + " " + url))
        self.log("Messaging Agent completed")

    def alert(self, opportunity: Opportunity) -> None:
        """Legacy helper: send a basic alert without LLM copywriting."""
        text = (
            f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f} — "
            f"{opportunity.deal.url}"
        )
        self.push(text)
