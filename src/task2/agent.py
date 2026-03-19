"""
Claude-powered agent that completes Tripletex accounting tasks via tool use.
"""
import base64
import json
import httpx
import anthropic

CLIENT = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

SYSTEM = """You are an expert accounting assistant that completes tasks in Tripletex, a Norwegian accounting SaaS.

You will receive a task prompt (possibly in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French).
Use the `tripletex` tool to make REST API calls and complete the task precisely.

IMPORTANT: Start making API calls IMMEDIATELY on your first response. Do not explain your plan first — just call tools.

Tripletex API conventions:
- Base URL is provided per-request — always use it exactly as given.
- Auth: HTTP Basic with username="0" and password=session_token (already handled by the tool).
- GET to fetch/list resources, POST to create, PUT to update, DELETE to remove.
- Most POST/PUT bodies are JSON. Responses wrap data in {"value": ...}.
- Dates: "yyyy-MM-dd". Employee numbers, customer numbers are auto-assigned (don't set them).
- Common endpoints:
    GET  /employee                    list employees
    POST /employee                    create employee {"firstName","lastName","email"(opt)}
    PUT  /employee/{id}               update employee
    GET  /customer                    list customers (?organizationNumber=X to find by org no.)
    POST /customer                    create customer {"name","organizationNumber"(opt),"email"(opt),"phoneNumber"(opt)}
    GET  /product                     list products
    POST /product                     create product {"name","number":"NNNN"(opt),"price":AMOUNT,"vatType":{"id":N}}
    GET  /order                       list orders
    POST /order                       create order {"customer":{"id":N},"orderDate":"yyyy-MM-dd","orderLines":[{"description":"...","unitPriceExcludingVatCurrency":AMOUNT,"vatType":{"id":N},"count":1}]}
    POST /invoice                     create invoice from order {"id":N,"invoiceDate":"yyyy-MM-dd","sendToCustomer":true/false}
                                      OR directly: {"customer":{"id":N},"invoiceDate":"yyyy-MM-dd","invoiceLines":[...]}
    PUT  /invoice/{id}/:pay           register payment — params: paymentDate=YYYY-MM-DD, paidAmount=AMOUNT, paymentTypeId=N
    GET  /payment/type                list payment types (use first available id for cash/bank)
    POST /invoice/{id}/:createCreditNote  create credit note
    POST /travelExpense               create travel expense report
    DELETE /travelExpense/{id}        delete travel expense report
    POST /project                     create project {"name":"...","customer":{"id":N}(opt),"projectManager":{"id":N}(opt)}
    GET  /department                  list departments
    POST /department                  create department {"name":"..."}
    GET  /ledger/vatType              list VAT types (id=3 is typically "No VAT/0%")

- Invoice creation workflow:
    1. GET /customer?organizationNumber=X&count=1 to find customer id
    2. GET /ledger/vatType to find correct VAT type id (0% VAT has type with "0" or "Ingen" in name)
    3. POST /order with customer id, today's date, and orderLines (unitPriceExcludingVatCurrency for the amount)
    4. POST /invoice with {"id": order_id, "invoiceDate": "yyyy-MM-dd", "sendToCustomer": true}

- Payment registration workflow (when task says "register payment" for existing invoice):
    1. GET /customer?organizationNumber=X&count=1 to find customer id
    2. GET /invoice?customerId=CUSTOMER_ID&invoiceStatus=OPEN to find unpaid invoices
    3. GET /payment/type to get valid payment type IDs
    4. PUT /invoice/{id}/:pay with params paymentDate=TODAY, paidAmount=AMOUNT, paymentTypeId=FIRST_ID

- VAT type rules:
  - If the task says "without VAT" / "uten MVA" / "uten mva" / "ex VAT" / "sem IVA" / "sans TVA" / "ohne MwSt" as the PRODUCT'S VAT type → use 0% vatType
  - BUT if the task says "price is X without VAT, at Y% rate" → the price is excl. VAT and the VAT rate is Y% (NOT 0%)
  - When a specific VAT rate is mentioned (e.g. "25%" or "standard rate") → use that rate, not 0%
  - GET /ledger/vatType first to find correct id by matching the percentage
- Always use today's date (2026-03-19) for invoiceDate/orderDate unless told otherwise.
- When looking up a customer by org number, use GET /customer?organizationNumber=X&count=1 and take response.value[0].id.
- For employee lookup by email: GET /employee?email=X&count=1.
- If a GET returns 403 or empty, the data may not exist yet — create it.

Always complete the full task before stopping. If a task requires multiple steps (e.g. create customer then create invoice for that customer), do them all.
When finished, stop calling tools and output nothing — the caller handles the response.
"""

TOOLS = [
    {
        "name": "tripletex",
        "description": "Make an HTTP request to the Tripletex API via the competition proxy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "description": "HTTP method"
                },
                "path": {
                    "type": "string",
                    "description": "API path, e.g. /employee or /invoice/42/:pay"
                },
                "body": {
                    "type": "object",
                    "description": "JSON request body for POST/PUT (omit for GET/DELETE)"
                },
                "params": {
                    "type": "object",
                    "description": "Query string parameters as key/value pairs (optional)"
                }
            },
            "required": ["method", "path"]
        }
    }
]


def _call_tripletex(base_url: str, session_token: str, method: str, path: str,
                    body: dict | None, params: dict | None) -> str:
    url = base_url.rstrip("/") + path
    # Tripletex v2 uses HTTP Basic auth: username="0", password=session_token
    basic = base64.b64encode(f"0:{session_token}".encode()).decode()
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30, verify=False) as client:
        resp = client.request(
            method=method,
            url=url,
            headers=headers,
            json=body,
            params=params,
        )
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    return json.dumps({"status_code": resp.status_code, "body": data}, ensure_ascii=False)


def run_agent(prompt: str, tripletex_base_url: str, session_token: str,
              attachments: list[dict] | None = None) -> None:
    """
    Run the accounting agent. Calls Tripletex until the task is complete.
    attachments: list of {"type": "base64", "media_type": "...", "data": "..."} for PDFs/images.
    """
    user_content: list = [{"type": "text", "text": prompt}]
    if attachments:
        for att in attachments:
            user_content.append(att)

    messages = [{"role": "user", "content": user_content}]

    while True:
        response = CLIENT.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input
            result = _call_tripletex(
                base_url=tripletex_base_url,
                session_token=session_token,
                method=inp["method"],
                path=inp["path"],
                body=inp.get("body"),
                params=inp.get("params"),
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})
