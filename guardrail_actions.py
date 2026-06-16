from nemoguardrails.actions import action

# salary, personal records, and anything HR-related
HR_PII_KEYWORDS: list[str] = [
    "salary", "salaries", "pay", "payroll",
    "compensation", "remuneration",
    "wage", "wages", "bonus",
    "personal data", "employee record",
    "date of birth", "social security", "ssn", "national id",
    # German equivalents carried over from the original keyword list
    "gehalt", "lohn", "adresse", "telefonnummer",
]

# passwords, tokens, keys, and anything that could grant system access
IT_SECURITY_KEYWORDS: list[str] = [
    "password", "passwort", "passphrase",
    "api key", "apikey", "api_key",
    "token", "auth token", "access token", "bearer token",
    "secret key", "private key", "secret",
    "ssh key", "encryption key",
    "credentials", "credential",
    "rfid", "zugangscode", "access code", "pin code",
]


@action(name="detect_sensitive_input")
async def detect_sensitive_input(user_message: str) -> str:
    # IT security is checked first because it's the higher-severity category
    text = user_message.lower()
    if any(kw in text for kw in IT_SECURITY_KEYWORDS):
        return "it_security"
    if any(kw in text for kw in HR_PII_KEYWORDS):
        return "hr_pii"
    return "clean"


@action(name="detect_sensitive_output")
async def detect_sensitive_output(bot_message: str) -> str:
    # catches cases where a sensitive term slipped through in a retrieved document chunk
    text = bot_message.lower()
    if any(kw in text for kw in IT_SECURITY_KEYWORDS):
        return "it_security"
    if any(kw in text for kw in HR_PII_KEYWORDS):
        return "hr_pii"
    return "clean"
