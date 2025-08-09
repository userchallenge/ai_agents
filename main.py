import os
import argparse
from datetime import datetime, date
from dateutil.parser import parse as parse_date
from typing import Optional, List

import instructor
from atomic_agents.agents.atomic_agent import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.context.system_prompt_generator import (
    SystemPromptGenerator,
    BaseDynamicContextProvider,
)
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


# ============================================================================
# EMAIL CATEGORIZATION AGENT COMPONENTS
# ============================================================================


class EmailCategorizationResult(BaseIOSchema):
    """Email categorization result."""

    date: Optional[str] = Field(
        default=None, description="Email date in ISO format or null if not found"
    )
    subject: Optional[str] = Field(
        default=None, description="Email subject or null if not found"
    )
    sender: Optional[str] = Field(
        default=None, description="Email sender address or null if not found"
    )
    body: Optional[str] = Field(
        default=None, description="Email body text or null if not found"
    )
    categories: List[str] = Field(
        default_factory=list,
        description="List of applicable categories: Friends, Family, Suppliers, Job Applications, Events",
    )


class EmailContextProvider(BaseDynamicContextProvider):
    """Context provider for email categorization with known contacts and domains."""

    def __init__(
        self,
        known_family_senders: Optional[List[str]] = None,
        known_friends_senders: Optional[List[str]] = None,
        last_name: Optional[str] = None,
        supplier_domains: Optional[List[str]] = None,
        extra_keywords: Optional[List[str]] = None,
    ):
        self.title = "Email Context"  # Required by atomic-agents framework
        self.known_family_senders = known_family_senders or []
        self.known_friends_senders = known_friends_senders or []
        self.last_name = last_name
        self.supplier_domains = supplier_domains or []
        self.extra_keywords = extra_keywords or []

    def get_info(self) -> str:
        """Return human-readable context information."""
        context_parts = []

        if self.known_family_senders:
            context_parts.append(
                f"Known family senders: {', '.join(self.known_family_senders)}"
            )

        if self.known_friends_senders:
            context_parts.append(
                f"Known friends senders: {', '.join(self.known_friends_senders)}"
            )

        if self.last_name:
            context_parts.append(f"User last name: {self.last_name}")

        if self.supplier_domains:
            context_parts.append(
                f"Known supplier domains: {', '.join(self.supplier_domains)}"
            )

        if self.extra_keywords:
            context_parts.append(
                f"Additional keywords: {', '.join(self.extra_keywords)}"
            )

        return (
            "\n".join(context_parts)
            if context_parts
            else "No specific context provided."
        )


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

# Define the order extraction agent
order_extractor_agent = AtomicAgent[EmailInputSchema, OrderExtractionResult](
    config=AgentConfig(
        client=client,
        model="gemini-2.0-flash",
        system_prompt_generator=system_prompt_generator,
        input_schema=EmailInputSchema,
        output_schema=OrderExtractionResult,
    )
)

# ============================================================================
# EMAIL CATEGORIZATION AGENT SETUP
# ============================================================================

# Email categorization system prompt
categorization_prompt = SystemPromptGenerator(
    background=[
        "You are an email categorization assistant that extracts fields and classifies emails.",
        "Apply any provided dynamic context about known contacts and domains before analyzing.",
    ],
    steps=[
        "First, apply the injected context about known family senders, friends, supplier domains, and keywords.",
        "Extract the date, subject, sender, and body from the email data.",
        "Classify the email into zero or more categories: Friends, Family, Suppliers, Job Applications, Events.",
        "An email may have multiple categories or none at all.",
        "If uncertain about categorization, leave the categories list empty rather than guessing.",
    ],
    output_instructions=[
        "Return extracted fields (date, subject, sender, body) and a categories array.",
        "Use null for any field that cannot be reliably extracted.",
        "Categories must be exactly: Friends, Family, Suppliers, Job Applications, Events.",
        "Return empty array for categories if uncertain.",
    ],
)

# Create context provider with sample data (you can customize this)
email_context = EmailContextProvider(
    known_family_senders=["mom@example.com", "dad@example.com"],
    known_friends_senders=["john@gmail.com", "sarah@yahoo.com"],
    last_name="Wahlström",
    supplier_domains=["bangerhead.se", "amazon.se", "madinter.com"],
    extra_keywords=["meeting", "birthday", "interview"],
)

# Define the categorization agent
categorization_agent = AtomicAgent[EmailInputSchema, EmailCategorizationResult](
    config=AgentConfig(
        client=client,
        model="gemini-2.0-flash",
        system_prompt_generator=categorization_prompt,
        input_schema=EmailInputSchema,
        output_schema=EmailCategorizationResult,
    )
)

# Register context provider
categorization_agent.register_context_provider("email_context", email_context)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Email processing with order extraction and/or categorization"
    )
    parser.add_argument(
        "--orders", action="store_true", help="Run order extraction agent"
    )
    parser.add_argument(
        "--categories", action="store_true", help="Run email categorization agent"
    )
    parser.add_argument(
        "--query", default="orderbekräftelse", help="Gmail search query"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of emails to process",
    )

    args = parser.parse_args()

    # Default to both if no specific agent is selected
    if not args.orders and not args.categories:
        args.orders = True
        args.categories = True

    print(f"Starting email processing...")
    print(f"🔍 Query: '{args.query}'")
    print(f"📊 Max results: {args.max_results}")
    print(
        f"🤖 Agents: {'Orders ' if args.orders else ''}{'Categories ' if args.categories else ''}"
    )

    # Get email data (using the query from args)
    email_data = search_emails(
        service, query=args.query, max_results=args.max_results, date_format="datetime"
    )

    print(f"✅ Found {len(email_data)} emails to process")

    if not email_data:
        print("❌ No emails found matching the query")
        return

    # Process each email in the search results
    for i, email in enumerate(email_data, 1):
        print(f"\n{'='*80}")
        print(f"📧 Processing Email {i}/{len(email_data)}")
        print(f"{'='*80}")
        print(f"Subject: {email['subject'][:80]}...")
        print(f"Sender: {email['sender'][:50]}...")
        print(f"Date: {email['date']}")

        # Create analysis request for this email
        analysis_request = EmailInputSchema(
            sender=email["sender"],
            subject=email["subject"],
            body=email["body_text"],
            date=email["date"],
        )

        # Run order extraction if requested
        if args.orders:
            print(f"\n--- 🛒 Order Analysis (Email {i}) ---")
            try:
                order_extractor_agent.reset_history()
                order_result = order_extractor_agent.run(analysis_request)

                print(
                    f"Is Order: {'✅' if order_result.is_order else '❌'} {order_result.is_order}"
                )
                print(f"Supplier: {order_result.supplier}")
                print(f"Total Amount: {order_result.total_amount}")
                print(f"Currency: {order_result.currency}")
                print(f"Confidence: {order_result.confidence:.2f}")
                print(f"Reasoning: {order_result.reasoning}")

                if order_result.order_date is None:
                    print("Order Date: ❌ NOT FOUND")
                else:
                    print(f"Order Date: ✅ {order_result.order_date}")

            except Exception as e:
                print(f"❌ Order analysis failed: {e}")

        # Run categorization if requested
        if args.categories:
            print(f"\n--- 🏷️ Categorization Analysis (Email {i}) ---")
            try:
                categorization_agent.reset_history()
                cat_result = categorization_agent.run(analysis_request)

                print(f"Date: {cat_result.date or '❌ NOT FOUND'}")
                print(f"Subject: {cat_result.subject or '❌ NOT FOUND'}")
                print(f"Sender: {cat_result.sender or '❌ NOT FOUND'}")
                print(f"Body: {'✅ Found' if cat_result.body else '❌ NOT FOUND'}")

                if cat_result.categories:
                    print(f"Categories: ✅ {', '.join(cat_result.categories)}")
                else:
                    print("Categories: ❌ None identified")

            except Exception as e:
                print(f"❌ Categorization failed: {e}")

    print(f"\n{'='*80}")
    print(f"✅ Completed processing {len(email_data)} emails")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
