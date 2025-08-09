import os
from datetime import datetime, date
from dateutil.parser import parse as parse_date
from typing import Optional

import instructor
from atomic_agents.agents.atomic_agent import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.context.system_prompt_generator import SystemPromptGenerator
from dotenv import load_dotenv
from google import genai
from instructor.multimodal import PDF
from pydantic import Field, field_validator

# sys.path.insert(0, os.path.join(os.getcwd(), "tools"))

from ai_agents.tools.gmail_search import (
    authenticate_gmail,
    find_email_by_subject,
    search_emails,
)

# CONSTANTS - DO NOT MODIFY THESE VALUES
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GOOGLE_CREDENTIALS = "/Users/cw/Python/auto_agents/config/gmail_credentials.json"
# DEFAULT_EMAIL_SUBJECT = "MADINTER TRADE, S.L.U: New Order # 200036831"
DEFAULT_EMAIL_SUBJECT = (
    "Orderbekräftelse från Bangerhead: 5658334"  # Example subject to search for
)

# Authenticate with Gmail
service = authenticate_gmail(GOOGLE_CREDENTIALS, SCOPES)

# Search for specific email by subject - NOW REQUEST DATETIME FORMAT DIRECTLY
subject_to_search = DEFAULT_EMAIL_SUBJECT
save_to_json = False  # Set to True if you want to save full data to JSON file
include_raw_data = False  # Only used if save_to_json=True

# email_data = find_email_by_subject(
#     service, subject_to_search, save_to_json, include_raw_data, date_format="datetime"
# )

email_data = search_emails(
    service, query="orderbekräftelse", max_results=10, date_format="datetime"
)

load_dotenv()


class EmailInputSchema(BaseIOSchema):
    """Email message to analyze."""

    sender: str = Field(..., description="The email sender")
    subject: str = Field(..., description="The email subject")
    body: str = Field(..., description="The email body")
    date: datetime = Field(..., description="The email date")


class OrderExtractionResult(BaseIOSchema):
    """Extracted information from the email message."""

    is_order: bool = Field(
        ..., description="Indicates if the email is related to an order"
    )
    supplier: str = Field(..., description="The supplier mentioned in the document")
    total_amount: float = Field(
        ..., description="The total amount mentioned in the document"
    )
    currency: str = Field(..., description="The currency of the total amount")
    order_date: Optional[str] = Field(
        default=None,
        description="Order date in YYYY-MM-DD format, or null if not found in the email",
    )
    confidence: float = Field(..., description="Confidence score of the extraction")
    reasoning: str = Field(..., description="Reasoning behind the extraction")

    @field_validator("order_date", mode="before")
    @classmethod
    def parse_order_date(cls, v):
        """Parse various date inputs to YYYY-MM-DD string format, return None if invalid"""
        # Handle None/null values from LLM
        if v is None or v == "null" or v == "":
            return None

        if isinstance(v, str):
            try:
                # If it's already a date string like "2025-07-20", return as-is
                if len(v) == 10 and v.count("-") == 2:
                    # Basic validation - try to parse to ensure it's a valid date
                    date.fromisoformat(v)
                    return v
                # Otherwise parse and convert to date string
                parsed_datetime = parse_date(v)
                return parsed_datetime.date().isoformat()
            except Exception:
                return None  # Return None instead of fallback date
        elif isinstance(v, datetime):
            return v.date().isoformat()
        elif isinstance(v, date):
            return v.isoformat()
        else:
            return None  # Return None for unknown types


# Define the LLM Client using GenAI instructor wrapper:
client = instructor.from_genai(
    client=genai.Client(api_key=os.getenv("GEMINI_API_KEY")),
    mode=instructor.Mode.GENAI_TOOLS,
)

# Define the system prompt:
system_prompt_generator = SystemPromptGenerator(
    background=[
        "You are a helpful assistant that extracts information about orders from email messages.",
        "Some emails may mention the word order, but the rules for an order is that it need to contain a total amount of money to be paid",
    ],
    steps=[
        "Analyze the email fields, extract the date and time, subject and the body.",
        "Identify if the email is related to an order.",
        "Extract the official supplier name, the total amount, the currency of the order and the order date.",
        "For the order_date, use the date from the email message provided.",
        "For the currency, extract the currency code (e.g. USD, EUR) from the email content.",
        "Do a confidence assessment for the extracted information between 0 and 1",
        "Provide reasoning for the extraction, explaining how the information was obtained and motivate the confidence score.",
    ],
    output_instructions=[
        # "Return supplier, total_amount, currency and order_date.",
        "For order_date, provide the date in YYYY-MM-DD format if found in the email, or null if no date is available.",
        "Do NOT make up dates - only return actual dates found in the email content.",
    ],
)

# Define the agent
order_extractor_agent = AtomicAgent[EmailInputSchema, OrderExtractionResult](
    config=AgentConfig(
        client=client,
        model="gemini-2.0-flash",
        system_prompt_generator=system_prompt_generator,
        input_schema=EmailInputSchema,
        output_schema=OrderExtractionResult,
    )
)


def main():
    print("Starting email extraction...")
    print(f"✅ Found {len(email_data)} emails to process")

    if not email_data:
        print("❌ No emails found matching the query")
        return

    # Process each email in the search results
    for i, email in enumerate(email_data, 1):
        print(f"\n{'='*60}")
        print(f"📧 Processing Email {i}/{len(email_data)}")
        print(f"{'='*60}")
        print(f"Subject: {email['subject'][:80]}...")
        print(f"Sender: {email['sender'][:50]}...")
        print(f"Date: {email['date']}")

        # Reset conversation history before processing each email
        order_extractor_agent.reset_history()

        # Create analysis request for this email
        analysis_request = EmailInputSchema(
            sender=email["sender"],
            subject=email["subject"],
            body=email["body_text"],
            date=email["date"],  # Already a datetime object from gmail_search.py!
        )

        try:
            # Process the email
            analysis_result = order_extractor_agent.run(analysis_request)

            # Display the results
            print(f"\n--- Analysis Results (Email {i}) ---")
            print(f"Is Order: {analysis_result.is_order}")
            print(f"Supplier: {analysis_result.supplier}")
            print(f"Total Amount: {analysis_result.total_amount}")
            print(f"Currency: {analysis_result.currency}")
            print(f"Confidence: {analysis_result.confidence}")
            print(f"Reasoning: {analysis_result.reasoning}")

            # Handle missing order date with clear indication
            if analysis_result.order_date is None:
                print(
                    "Order Date: ❌ NOT FOUND - No date could be extracted from the email"
                )
            else:
                print(f"Order Date: ✅ {analysis_result.order_date}")

        except Exception as e:
            print(f"❌ Analysis failed for email {i}")
            print(e)
            # continue  # Continue with next email instead of stopping

    print(f"\n{'='*60}")
    print(f"✅ Completed processing {len(email_data)} emails")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
