from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for AI prompts"""
    system_prompt: str
    user_template: str
    examples: Optional[List[Dict[str, str]]] = None


class BankingPrompts:
    """
    Banking domain prompt templates for consistent AI responses
    """
    
    
    SYSTEM_PROMPT = """You are an expert AI assistant acting as a Senior Customer Support Specialist for a major financial institution. You have deep, comprehensive knowledge of all banking products, internal procedures, and customer resolution protocols.

Your role is to:
1. Analyze customer requests and complaints related to banking services.
2. Generate structured support tickets with proper categorization.
3. Formulate a detailed, step-by-step `resolutionSuggestion` that a junior support agent could follow to resolve the issue.
4. ONLY respond to banking and financial services related queries.
5. Always respond in the exact JSON format specified.

CRITICAL RULES:
- REJECT any non-banking/financial queries immediately.
- Always maintain professional banking terminology.
- Ensure data privacy and security in responses.
- Focus on banking services: accounts, cards, loans, investments, payments, mobile banking, etc.
- Assign appropriate severity levels based on impact and urgency.
- Route to correct banking departments.
- Detect preferred communication method from customer text (e.g., "please call me", "send email", "text me", etc.).

Response must be in this exact JSON format:
{
  "title": "Brief title summarizing the issue",
  "description": "Concise summary of the customer's issue (not the original text)",
  "originalInput": "Complete original customer text",
  "severity": "Low|Medium|High|Critical",
  "typeOfTicket": "Complaint|Inquiry|Assistance",
  "impactedDepartment": "Account Management|Credit Cards|Loans and Mortgages|Investment Services|Customer Service|Technical Support|Fraud Prevention|Compliance|Wire Transfers|Mobile Banking",
  "impactedService": "Specific service affected",
  "customerName": "Customer name if provided, otherwise an empty string \"\"",
  "customerId": "Customer's unique national ID or account identifier if provided, otherwise an empty string \"\"",
  "resolutionSuggestion": "A detailed, step-by-step action plan for an internal support agent. It should be professional, comprehensive, and provide clear instructions to resolve the issue or escalate it appropriately. Include specific internal actions where applicable.",
  "preferredCommunication": "Based on user input (e.g., 'call me', 'email me'). If no preference is specified, this MUST be an empty string \"\"",
  "id": "Original ticket ID from the source system"
}

For non-banking queries, respond with:
{
  "error": "Non-banking query rejected",
  "message": "I can only assist with banking and financial services related inquiries.",
  "isBankingRelated": false,
  "confidenceScore": 0.0
}"""

    USER_PROMPT_TEMPLATE = """Customer Request:
{customerText}

Customer Information:
- ID: {id}
- Customer Name: {customerName}
- Customer ID: {customerId}

Please analyze this request and generate a structured support ticket with the provided ID if it's banking-related, or reject it if it's not."""

    VALIDATION_PROMPT = """Analyze the following customer text and determine if it's related to banking or financial services.

Customer Text: "{customerText}"

Banking/Financial Topics Include:
- Bank accounts (checking, savings, business)
- Credit/debit cards
- Loans, mortgages, credit
- Investments, trading, portfolios  
- Payments, transfers, wires
- Mobile/online banking apps
- ATMs, branches, customer service
- Fraud, security, identity theft
- Banking fees, statements, reports
- Regulatory compliance issues

Respond with only "BANKING" if related to financial services, or "NOT_BANKING" if unrelated."""

    @classmethod
    def get_ticket_generation_prompt(cls, customerText: str, id: str, 
                                   customerName: Optional[str] = None, 
                                   customerId: Optional[str] = None) -> str:
        """
        Generate complete prompt for ticket generation
        """
        return cls.USER_PROMPT_TEMPLATE.format(
            customerText=customerText,
            id=id,
            customerName=customerName or "", # MODIFIED: Default to empty string
            customerId=customerId or ""   # MODIFIED: Default to empty string
        )
    
    @classmethod
    def get_validation_prompt(cls, customerText: str) -> str:
        """
        Generate prompt for banking domain validation
        """
        return cls.VALIDATION_PROMPT.format(customerText=customerText)

    
    EXAMPLES = [
        {
            "input": "I can't access my mobile banking app, it keeps crashing when I try to log in. Please call me to help resolve this.",
            "output": """{
  "title": "Mobile Banking App Login Issue",
  "description": "Customer experiencing mobile banking application crashes during login attempts, preventing access to online banking services.",
  "severity": "High",
  "typeOfTicket": "Assistance",
  "impactedDepartment": "Technical Support",
  "impactedService": "Mobile Banking Application",
  "customerName": "",
  "customerId": "",
  "resolutionSuggestion": "1. Verify customer identity using the standard security Q&A protocol. 2. Advise customer to perform basic troubleshooting: a) Ensure the app is updated to the latest version. b) Restart their mobile device. c) Clear the app's cache. 3. If the issue persists, create a Tier 2 technical support ticket using the 'Mobile Banking' template in the CRM. 4. Document the customer's device model, operating system version, and the exact time of the crash in the ticket. 5. Inform the customer that the issue has been escalated and provide the new ticket number for their reference.",
  "preferredCommunication": "Phone",
  "confidenceScore": 0.95,
  "id": "Original ticket ID from the source system"
}"""
        },
        {
            "input": "There's an unauthorized charge of $500 on my credit card statement. Please email me the dispute form.",
            "output": """{
  "title": "Unauthorized Credit Card Transaction",
  "description": "Customer reports unauthorized charge of $500 appearing on credit card statement, potential fraud case requiring immediate investigation.",
  "severity": "Critical",
  "typeOfTicket": "Complaint",
  "impactedDepartment": "Fraud Prevention",
  "impactedService": "Credit Card Services",
  "customerName": "",
  "customerId": "",
  "resolutionSuggestion": "1. Reassure the customer and verify their identity using enhanced security protocols for fraud cases. 2. Immediately place a temporary block on the affected credit card via the Card Services portal to prevent further unauthorized charges. 3. Initiate a formal fraud investigation by filing a 'Dispute Claim - Unauthorized Transaction' form. 4. Inform the customer of the generated case number and advise that a new card will be issued and mailed within 5-7 business days. 5. Set a follow-up reminder to check the case status in 48 hours.",
  "preferredCommunication": "Email",
  "confidenceScore": 0.98,
  "id": "Original ticket ID from the source system"
}"""
        },
        {
            "input": "My savings account balance seems incorrect. The last deposit from Friday isn't showing up yet.",
            "output": """{
  "title": "Missing Deposit in Savings Account",
  "description": "Customer reports that a recent deposit made on Friday is not reflected in their savings account balance.",
  "severity": "Medium",
  "typeOfTicket": "Inquiry",
  "impactedDepartment": "Account Management",
  "impactedService": "Savings Account",
  "customerName": "",
  "customerId": "",
  "resolutionSuggestion": "1. Verify the customer's identity and access their savings account details. 2. Check the transaction history for the specified date to locate the deposit. 3. Inform the customer about standard deposit processing times (e.g., 1-2 business days). 4. If the deposit is not found and the processing time has elapsed, ask the customer for more details (e.g., deposit method, amount). 5. If necessary, initiate a transaction trace request with the Deposits department.",
  "preferredCommunication": "",
  "confidenceScore": 0.96,
  "id": "Original ticket ID from the source system"
}"""
        },
        {
            "input": "What's the weather like today?",
            "output": """{
  "error": "Non-banking query rejected",
  "message": "I can only assist with banking and financial services related inquiries.",
  "isBankingRelated": false,
  "confidenceScore": 0.0
}"""
        }
    ]

    # ... (The rest of the class methods remain unchanged) ...
    @classmethod
    def get_ticket_type_definitions(cls) -> Dict[str, str]:
        """
        Get definitions for each ticket type.
        """
        return {
            "Inquiry": "The customer is asking for information (e.g., 'What is my balance?', 'What are your hours?').",
            "Complaint": "The customer is reporting a problem, error, or dissatisfaction with a service (e.g., 'There is a wrong charge on my account.', 'Your app is broken.').",
            "Assistance": "The customer needs help performing an action or process (e.g., 'How do I transfer money?', 'Please help me reset my password.')."
        }

    @classmethod
    def get_severity_guidelines(cls) -> Dict[str, str]:
        """
        Get severity level assignment guidelines
        """
        return {
            "Critical": "System outages, fraud, security breaches, money lost/stolen, account compromised",
            "High": "Service disruptions affecting access, failed transactions, urgent account issues",
            "Medium": "Feature not working properly, minor service issues, general inquiries needing quick resolution",
            "Low": "Information requests, minor complaints, feature requests, general feedback"
        }
    
    @classmethod
    def get_department_routing(cls) -> Dict[str, List[str]]:
        """
        Get department routing keywords
        """
        return {
            "Account Management": [
                "account", "balance", "statement", "deposit", "withdrawal", 
                "account closure", "account opening", "account type"
            ],
            "Credit Cards": [
                "credit card", "card payment", "card limit", "card activation",
                "card replacement", "card fees", "reward points", "cashback"
            ],
            "Loans and Mortgages": [
                "loan", "mortgage", "refinance", "payment", "interest rate",
                "loan application", "loan approval", "loan balance"
            ],
            "Investment Services": [
                "investment", "portfolio", "stocks", "bonds", "mutual funds",
                "trading", "market", "advisor", "retirement"
            ],
            "Technical Support": [
                "app", "website", "login", "password", "system", "error",
                "mobile banking", "online banking", "technical issue"
            ],
            "Fraud Prevention": [
                "fraud", "unauthorized", "suspicious", "stolen", "hacked",
                "identity theft", "security breach", "scam"
            ],
            "Wire Transfers": [
                "wire transfer", "international transfer", "transfer money",
                "transfer fee", "transfer limit", "transfer status"
            ],
            "Customer Service": [
                "complaint", "service", "branch", "representative", "hours",
                "location", "appointment", "general inquiry"
            ]
        }

    @classmethod
    def get_banking_keywords(cls) -> List[str]:
        """
        Get comprehensive list of banking-related keywords
        """
        return [
            # Account types
            "checking", "savings", "account", "balance", "deposit", "withdrawal",
            "overdraft", "statement", "transaction", "transfer",
            
            # Cards
            "credit card", "debit card", "card", "payment", "charge", "limit",
            "activation", "replacement", "PIN", "CVV", "chip",
            
            # Loans & Credit
            "loan", "mortgage", "credit", "interest", "rate", "payment",
            "refinance", "approval", "application", "debt",
            
            # Digital Banking
            "mobile banking", "online banking", "app", "website", "login",
            "password", "authentication", "digital wallet",
            
            # Services
            "wire transfer", "ATM", "branch", "teller", "investment",
            "portfolio", "trading", "insurance", "safe deposit",
            
            # Issues
            "fraud", "unauthorized", "dispute", "complaint", "error",
            "problem", "issue", "help", "support", "customer service",
            
            # Financial terms
            "bank", "banking", "financial", "money", "cash", "funds",
            "currency", "exchange", "fee", "charge", "commission"
        ]