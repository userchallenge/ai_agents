import os
import pickle
import json
import base64
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from bs4 import BeautifulSoup


def authenticate_gmail(google_credentials, scopes):
    """
    Authenticate with Gmail API using OAuth2 flow.

    Args:
        google_credentials (str): Path to the Google OAuth2 credentials JSON file
        scopes (list): List of Gmail API scopes to request access for

    Returns:
        googleapiclient.discovery.Resource: Authenticated Gmail API service object

    Note:
        Creates/updates 'token.pickle' file to store authentication tokens for reuse.
    """
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(google_credentials, scopes)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("gmail", "v1", credentials=creds)


def decode_message_body(payload):
    """
    Extract and decode email body text from Gmail message payload, handling HTML content.

    Args:
        payload (dict): Gmail message payload containing body data and parts

    Returns:
        str: Decoded and cleaned text content from the email body

    Note:
        - Recursively processes multipart messages
        - Converts HTML content to plain text using BeautifulSoup
        - Handles base64url decoding of message data
    """

    def extract_text_from_part(part):
        text = ""
        if "data" in part.get("body", {}):
            # Decode base64 content
            data = part["body"]["data"]
            decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")

            # Check if it's HTML and convert to readable text
            mime_type = part.get("mimeType", "")
            if "html" in mime_type:
                soup = BeautifulSoup(decoded, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
            else:
                text = decoded

        elif "parts" in part:
            # Handle multipart messages
            for subpart in part["parts"]:
                text += extract_text_from_part(subpart)
        return text

    return extract_text_from_part(payload)


def clean_message_data(message, include_data=False):
    """
    Clean Gmail message data by removing or replacing base64 data sections.

    Args:
        message (dict): Gmail message object containing payload data
        include_data (bool, optional): If True, keeps original data. If False,
                                     replaces with length indicator. Defaults to False.

    Returns:
        dict: Deep copy of message with data sections cleaned according to include_data flag

    Note:
        Recursively processes all parts and subparts of the message payload.
    """

    def process_part(part):
        if "body" in part and "data" in part["body"]:
            if not include_data:
                data_length = len(part["body"]["data"])
                part["body"]["data"] = f"data ({data_length} characters) not presented"

        if "parts" in part:
            for subpart in part["parts"]:
                process_part(subpart)

    cleaned_message = json.loads(json.dumps(message))  # Deep copy
    process_part(cleaned_message["payload"])
    return cleaned_message


def extract_sender_email(from_header):
    """
    Extract email address from Gmail From header field.

    Args:
        from_header (str): Raw From header string (e.g., "Name <email@domain.com>")

    Returns:
        str: Extracted email address or original string if no angle brackets found

    Example:
        >>> extract_sender_email("John Doe <john@example.com>")
        'john@example.com'
        >>> extract_sender_email("simple@email.com")
        'simple@email.com'
    """
    if "<" in from_header and ">" in from_header:
        match = re.search(r"<(.+?)>", from_header)
        if match:
            return match.group(1)
    return from_header


def fetch_gmail_messages(service, query, max_results=10):
    """
    Fetch Gmail messages matching a query and return full message objects.
    
    Args:
        service (googleapiclient.discovery.Resource): Authenticated Gmail API service
        query (str): Gmail search query string
        max_results (int, optional): Maximum number of messages to fetch. Defaults to 10.
        
    Returns:
        list: List of full Gmail message objects from the API
        
    Note:
        This is a low-level helper that fetches raw Gmail message data.
        Use convert_message_to_simple_format() to get simplified format.
    """
    try:
        results = (
            service.users()
            .messages()
            .list(userId="me", q=query, maxResults=max_results)
            .execute()
        )
        messages = results.get("messages", [])
        
        full_messages = []
        for msg in messages:
            try:
                message = (
                    service.users().messages().get(userId="me", id=msg["id"]).execute()
                )
                full_messages.append(message)
            except Exception as e:
                print(f"Error getting message {msg['id']}: {e}")
                
        return full_messages
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []


def convert_message_to_simple_format(message):
    """
    Convert a Gmail message object to simplified format matching search_emails output.
    
    Args:
        message (dict): Full Gmail message object from API
        
    Returns:
        dict: Simplified message format containing:
            - id: Gmail message ID
            - subject: Email subject line
            - sender: From header value
            - date: Date in ISO format (YYYY-MM-DDTHH:MM:SS+TZ) or None if parsing fails
            - body_text: Full formatted body text content
            
    Note:
        This creates the same format as search_emails() for consistency.
    """
    try:
        headers = message["payload"].get("headers", [])

        subject = next(
            (h["value"] for h in headers if h["name"] == "Subject"), "No Subject"
        )
        sender = next(
            (h["value"] for h in headers if h["name"] == "From"), "No Sender"
        )
        date_raw = next((h["value"] for h in headers if h["name"] == "Date"), "No Date")
        
        # Convert date to ISO format
        try:
            from email.utils import parsedate_to_datetime
            if date_raw != "No Date":
                date_obj = parsedate_to_datetime(date_raw)
                date = date_obj.isoformat()
            else:
                date = None
        except:
            date = None

        # Get full body text
        body_text = decode_message_body(message["payload"])

        return {
            "id": message["id"],
            "subject": subject,
            "sender": sender,
            "date": date,
            "body_text": body_text,
        }
    except Exception as e:
        print(f"Error converting message to simple format: {e}")
        return None


def format_filename(message):
    """
    Create standardized filename for Gmail message JSON files.

    Args:
        message (dict): Gmail message object containing headers and metadata

    Returns:
        str: Formatted filename in format "email_YYMMDD:HHMM_<sender_email>.json"

    Example:
        >>> # For email from john@example.com sent on 2025-07-20 at 10:06
        >>> format_filename(message)
        'email_250720:1006_john@example.com.json'

    Note:
        - Date format is YYMMDD:HHMM (24-hour format)
        - Invalid filename characters in email address are replaced with underscores
        - Falls back to "000000:0000" if date parsing fails
    """
    headers = message["payload"].get("headers", [])

    # Get date and sender
    date_header = next((h["value"] for h in headers if h["name"] == "Date"), "")
    from_header = next((h["value"] for h in headers if h["name"] == "From"), "")

    # Parse date to YYMMDD:HHMM format
    try:
        # Gmail date format parsing
        from email.utils import parsedate_to_datetime

        date_obj = parsedate_to_datetime(date_header)
        date_str = date_obj.strftime("%y%m%d:%H%M")
    except:
        date_str = "000000:0000"

    # Extract sender email
    sender_email = extract_sender_email(from_header)

    # Clean sender email for filename (remove invalid characters)
    sender_clean = re.sub(r'[<>:"/\\|?*]', "_", sender_email)

    return f"email_{date_str}_{sender_clean}.json"


def find_email_by_subject(service, subject, save_to_json=False, include_raw_data=False):
    """
    Find the first email matching a subject line and return simplified format.

    Args:
        service (googleapiclient.discovery.Resource): Authenticated Gmail API service
        subject (str): Email subject line to search for (exact match)
        save_to_json (bool, optional): Whether to save full message to JSON file. Defaults to False.
        include_raw_data (bool, optional): Whether to include raw base64 data in JSON (only if save_to_json=True).
                                         Defaults to False.

    Returns:
        dict or None: Simplified message format matching search_emails output:
            - id: Gmail message ID
            - subject: Email subject line
            - sender: From header value
            - date: Date in ISO format (YYYY-MM-DDTHH:MM:SS+TZ) or None if parsing fails
            - body_text: Full formatted body text content
            Returns None if no matching email found.

    Note:
        - Searches using Gmail query format: subject:"<subject>"
        - Only returns the first match if multiple emails have same subject
        - Returns same format as search_emails() for consistency
        - Optionally saves full message data to JSON file if save_to_json=True
    """
    # Use helper method to fetch messages
    messages = fetch_gmail_messages(service, f'subject:"{subject}"', max_results=1)
    
    if messages:
        message = messages[0]  # Get first match
        
        # Convert to simplified format using helper method
        simple_format = convert_message_to_simple_format(message)
        
        # Optionally save to JSON file
        if save_to_json and simple_format:
            # Extract readable text for JSON
            email_text = decode_message_body(message["payload"])
            
            # Clean message data based on parameter
            cleaned_message = clean_message_data(message, include_raw_data)
            cleaned_message["decoded_body_text"] = email_text
            
            # Create formatted filename and save
            filename = format_filename(message)
            output_file = f"output_files/{filename}"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_message, f, indent=2, ensure_ascii=False)
                
            # Add file info to return data (for backward compatibility if needed)
            simple_format["_saved_to"] = output_file
            simple_format["_filename"] = filename
        
        return simple_format
    
    return None


def search_emails(service, query, max_results=10):
    """
    Search emails using Gmail query syntax and return summary information.

    Args:
        service (googleapiclient.discovery.Resource): Authenticated Gmail API service
        query (str): Gmail search query string (e.g., "invoice", "from:example.com")
        max_results (int, optional): Maximum number of emails to return. Defaults to 10.

    Returns:
        list: List of dictionaries, each containing:
            - id: Gmail message ID
            - subject: Email subject line
            - sender: From header value
            - date: Date in ISO format (YYYY-MM-DDTHH:MM:SS+TZ) or None if parsing fails
            - body_text: Full formatted body text content

    Example:
        >>> search_emails(service, "invoice", 5)
        [{'id': '12345', 'subject': 'Invoice #123', 'sender': 'billing@company.com',
          'date': '2024-01-01T10:00:00+00:00', 'body_text': 'Thank you for your purchase...'}]

    Note:
        - Uses Gmail's advanced search syntax
        - Errors retrieving individual messages are logged but don't stop the search
        - Returns full formatted body text (HTML converted to plain text)
    """
    # Use helper method to fetch messages
    messages = fetch_gmail_messages(service, query, max_results)
    
    # Convert each message to simplified format using helper method
    email_list = []
    for message in messages:
        simple_format = convert_message_to_simple_format(message)
        if simple_format:
            email_list.append(simple_format)

    return email_list


def save_message_to_json(message, include_raw_data=False, output_dir="output_files"):
    """
    Save a Gmail message object to a JSON file with standardized formatting.

    Args:
        message (dict): Gmail message object from API
        include_raw_data (bool, optional): Whether to include raw base64 data in output.
                                         If False, replaces with size indicator. Defaults to False.
        output_dir (str, optional): Directory to save JSON files. Defaults to "output_files".

    Returns:
        dict: Dictionary containing:
            - message: Original Gmail message object
            - email_text: Decoded body text content
            - output_file: Full path to saved JSON file
            - filename: Generated filename only

    Example:
        >>> result = save_message_to_json(message, include_raw_data=False)
        >>> print(result['output_file'])
        'output_files/email_250720:1006_sender@example.com.json'

    Note:
        - Creates output directory if it doesn't exist
        - Filename format: email_YYMMDD:HHMM_<sender_email>.json
        - Adds 'decoded_body_text' field to the JSON for easy reading
        - Uses UTF-8 encoding with proper indentation
    """
    # Extract readable text
    email_text = decode_message_body(message["payload"])

    # Clean message data based on parameter
    cleaned_message = clean_message_data(message, include_raw_data)
    cleaned_message["decoded_body_text"] = email_text

    # Create formatted filename
    filename = format_filename(message)
    output_file = f"{output_dir}/{filename}"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_message, f, indent=2, ensure_ascii=False)

    return {
        "message": message,
        "email_text": email_text,
        "output_file": output_file,
        "filename": filename,
    }
