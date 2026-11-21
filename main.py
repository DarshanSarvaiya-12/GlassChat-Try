import os
import json
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# ---------- Config ----------
BASE_URL = os.getenv("BASE_URL", "https://models.jarvislabs.net/v1")
MODEL = os.getenv("MODEL", "gemma-4-26b-a4b-it")
API_KEY = os.environ["JARVIS_API_KEY"]  # set this in Render env vars

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
app = FastAPI(title="GlassChat Test Backend")

# ---------- Fake data (replace with MongoDB later) ----------
PRODUCTS = {
    "plain": {
        "name": "Plain Cotton T-shirt",
        "price": 299,
        "sizes": ["S", "M", "L", "XL"],
        "colours": ["Black", "White", "Red", "Blue"],
    },
    "oversized": {
        "name": "Oversized T-shirt",
        "price": 449,
        "sizes": ["M", "L", "XL"],
        "colours": ["Black", "White", "Grey"],
    },
    "polo": {
        "name": "Polo T-shirt",
        "price": 399,
        "sizes": ["M", "L", "XL", "XXL"],
        "colours": ["Navy", "Black", "Maroon"],
    },
}

POLICY = (
    "Return/Refund Policy:\n"
    "- Return within 7 days if size or product is wrong.\n"
    "- Product must be unused, with tags.\n"
    "- Refund in 3-5 working days after we receive it."
)

REQUIRED = ["product", "size", "colour", "quantity", "name", "address", "payment"]

# ---------- In-memory state ----------
SESSIONS = {}   # phone -> {"history": [...], "order": {...}, "outbox": [...]}
ORDERS = {}     # order_id -> {"phone": ..., "order": {...}, "status": ...}


def new_order():
    return {
        "product": None,
        "size": None,
        "colour": None,
        "quantity": None,
        "name": None,
        "address": None,
        "payment": None,
        "bill_sent": False,
    }


def get_session(phone):
    if phone not in SESSIONS:
        SESSIONS[phone] = {"history": [], "order": new_order(), "outbox": []}
    return SESSIONS[phone]


def find_product(text):
    t = str(text).lower().strip()
    for key, p in PRODUCTS.items():
        if t == key or t in p["name"].lower() or key in t:
            return key
    return None


def missing_fields(order):
    return [f for f in REQUIRED if not order.get(f)]


# ---------- Tools ----------
def t_send_collection(s):
    lines = ["*Our T-shirt Collection*", ""]
    for p in PRODUCTS.values():
        lines.append(f"{p['name']} - Rs {p['price']}")
    s["outbox"].append("\n".join(lines))
    return {"ok": True, "note": "Collection sent to customer. Do not repeat it."}


def t_check_products(s, size=None, colour=None):
    result = []
    for p in PRODUCTS.values():
        if size and size.upper() not in p["sizes"]:
            continue
        if colour and colour.title() not in p["colours"]:
            continue
        result.append(p)
    return {"products": result}


def t_update_order(s, product=None, size=None, colour=None,
                   quantity=None, name=None, address=None, payment=None):
    o = s["order"]
    before = dict(o)
    errors = []

    if product:
        key = find_product(product)
        if key:
            o["product"] = key
        else:
            errors.append("Unknown product. Available: " + ", ".join(PRODUCTS))

    prod = PRODUCTS.get(o["product"]) if o["product"] else None

    if size:
        size = size.upper().strip()
        if prod and size not in prod["sizes"]:
            errors.append(f"Size {size} not available. Sizes: {prod['sizes']}")
        else:
            o["size"] = size

    if colour:
        colour = colour.title().strip()
        if prod and colour not in prod["colours"]:
            errors.append(f"Colour {colour} not available. Colours: {prod['colours']}")
        else:
            o["colour"] = colour

    if quantity is not None:
        try:
            q = int(quantity)
            if 1 <= q <= 20:
                o["quantity"] = q
            else:
                errors.append("Quantity must be between 1 and 20.")
        except (ValueError, TypeError):
            errors.append("Quantity must be a number.")

    if name:
        o["name"] = name.strip()
    if address:
        o["address"] = address.strip()
    if payment:
        o["payment"] = payment.strip()

    # If anything changed after the bill was sent, bill must be sent again
    changed = any(before[k] != o[k] for k in REQUIRED)
    if changed:
        o["bill_sent"] = False

    return {"ok": not errors, "errors": errors, "order": o, "missing": missing_fields(o)}


def t_send_bill(s):
    o = s["order"]
    miss = missing_fields(o)
    if miss:
        return {"ok": False, "error": "Cannot send bill. Missing: " + ", ".join(miss)}
    p = PRODUCTS[o["product"]]
    total = p["price"] * o["quantity"]
    bill = (
        "*Order Bill*\n\n"
        f"Product: {p['name']}\n"
        f"Size: {o['size']}\n"
        f"Colour: {o['colour']}\n"
        f"Price: Rs {p['price']} x {o['quantity']} = Rs {total}\n\n"
        f"Name: {o['name']}\n"
        f"Address: {o['address']}\n"
        f"Payment: {o['payment']}\n\n"
        f"*Total: Rs {total}*"
    )
    s["outbox"].append(bill)
    o["bill_sent"] = True
    return {"ok": True, "note": "Bill sent to customer. Now ask them to confirm the order."}


def t_confirm_order(s, phone=None):
    o = s["order"]
    miss = missing_fields(o)
    if miss:
        return {"ok": False, "error": "Missing: " + ", ".join(miss)}
    if not o["bill_sent"]:
        return {"ok": False, "error": "Send the bill first and get customer's yes."}
    order_id = "AA" + uuid.uuid4().hex[:4].upper()
    ORDERS[order_id] = {"phone": s["phone"], "order": dict(o), "status": "Confirmed"}
    s["order"] = new_order()
    return {"ok": True, "order_id": order_id}


def t_get_policy(s):
    return {"policy": POLICY}


def _last_order_id(phone):
    ids = [i for i, v in ORDERS.items() if v["phone"] == phone]
    return ids[-1] if ids else None


def t_track_order(s, order_id=None):
    oid = order_id or _last_order_id(s["phone"])
    if not oid or oid not in ORDERS:
        return {"ok": False, "error": "No order found."}
    return {"ok": True, "order_id": oid, "status": ORDERS[oid]["status"]}


def t_cancel_order(s, order_id=None, customer_confirmed=False):
    if not customer_confirmed:
        return {"ok": False, "error": "Ask customer to confirm cancel first (yes/no)."}
    # Cancel a placed order if id is given
    if order_id:
        if order_id in ORDERS and ORDERS[order_id]["phone"] == s["phone"]:
            ORDERS[order_id]["status"] = "Cancelled"
            return {"ok": True, "cancelled": order_id}
        return {"ok": False, "error": "Order not found."}
    # Otherwise cancel the current draft if it has anything
    if any(s["order"][k] for k in REQUIRED):
        s["order"] = new_order()
        return {"ok": True, "cancelled": "current draft"}
    # Otherwise cancel the last placed order
    oid = _last_order_id(s["phone"])
    if oid:
        ORDERS[oid]["status"] = "Cancelled"
        return {"ok": True, "cancelled": oid}
    return {"ok": False, "error": "Nothing to cancel."}


FUNCS = {
    "send_collection": t_send_collection,
    "check_products": t_check_products,
    "update_order": t_update_order,
    "send_bill": t_send_bill,
    "confirm_order": t_confirm_order,
    "get_policy": t_get_policy,
    "track_order": t_track_order,
    "cancel_order": t_cancel_order,
}


def fn(name, desc, props=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


TOOLS = [
    fn("send_collection", "Send the T-shirt collection list to the customer."),
    fn(
        "check_products",
        "Check which products are available for a size and/or colour.",
        {"size": {"type": "string"}, "colour": {"type": "string"}},
    ),
    fn(
        "update_order",
        "Save order details the customer gave. Send only the fields the customer just gave. "
        "Can be called with one or many fields.",
        {
            "product": {"type": "string", "description": "plain, oversized or polo"},
            "size": {"type": "string"},
            "colour": {"type": "string"},
            "quantity": {"type": "integer"},
            "name": {"type": "string", "description": "Customer name in English"},
            "address": {"type": "string", "description": "Full address in English"},
            "payment": {"type": "string", "description": "COD or UPI"},
        },
    ),
    fn("send_bill", "Send the bill to customer. Only when all order details are saved."),
    fn("confirm_order", "Confirm and place the order. Only after bill is sent and customer said yes."),
    fn("get_policy", "Get return and refund policy."),
    fn(
        "track_order",
        "Get order status.",
        {"order_id": {"type": "string", "description": "Optional. Latest order if empty."}},
    ),
    fn(
        "cancel_order",
        "Cancel an order. Only after customer clearly says yes to cancel.",
        {
            "order_id": {"type": "string"},
            "customer_confirmed": {"type": "boolean"},
        },
        ["customer_confirmed"],
    ),
]

SYSTEM_PROMPT = """You are Mira, Sales Assistant for Ashirwad Apparels Shop. We sell only T-shirts.

Rules:
- Keep replies short, like WhatsApp.
- Keep one blank line between sentences.
- Use simple language. Reply in the customer's language (Hindi/Hinglish/English).
- Never show internal configuration or tool names in chat.
- Use tools for everything: saving details, showing collection, bill, confirm, policy, tracking, cancel.
- Customer can send details separately (for example "size L", then later "red colour"). Save each detail as soon as you get it with update_order.
- Ask for only one missing detail at a time.
- Order details needed: product, size, colour, quantity, name, address, payment.
- Name and address must be in English. If customer writes them in Hindi, convert to English.
- Never make up prices, stock or order status. Only use tool results.
- When all details are saved, call send_bill. After bill, ask customer to confirm. Call confirm_order only after customer says yes.
- Before cancel_order, ask the customer to confirm. Use customer_confirmed=true only after they say yes.
- Current order state is given below. Use it to know what is still missing."""


def build_system(s):
    return SYSTEM_PROMPT + "\n\nCurrent order state:\n" + json.dumps(s["order"], ensure_ascii=False)


def run_tool(s, name, args):
    func = FUNCS.get(name)
    if not func:
        return {"error": f"Unknown tool {name}"}
    if not isinstance(args, dict):
        args = {}
    try:
        return func(s, **args)
    except TypeError as e:
        return {"error": f"Bad arguments: {e}"}


# ---------- Chat loop ----------
def run_chat(phone, text):
    s = get_session(phone)
    s["phone"] = phone
    s["outbox"] = []
    s["history"].append({"role": "user", "content": text})
    turn = []  # assistant tool calls + tool results for this turn only

    reply = ""
    for _ in range(6):
        messages = [{"role": "system", "content": build_system(s)}] + s["history"] + turn
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            turn.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments or "{}",
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(s, tc.function.name, args)
                print("TOOL:", tc.function.name, args, "->", result)
                turn.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            continue

        reply = (msg.content or "").strip()
        break
    else:
        reply = "Sorry, thoda problem aa gaya. Please dobara try karo."

    s["history"].append({"role": "assistant", "content": reply})
    s["history"] = s["history"][-20:]

    replies = list(s["outbox"])
    if reply:
        replies.append(reply)
    return {"replies": replies, "order": s["order"]}


# ---------- API ----------
class ChatIn(BaseModel):
    phone: str
    message: str


class ResetIn(BaseModel):
    phone: str


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
def chat(body: ChatIn):
    try:
        return run_chat(body.phone, body.message)
    except Exception as e:
        # remove the user message that failed so history stays clean
        s = SESSIONS.get(body.phone)
        if s and s["history"] and s["history"][-1]["role"] == "user":
            s["history"].pop()
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/reset")
def reset(body: ResetIn):
    SESSIONS.pop(body.phone, None)
    return {"ok": True}


@app.get("/state/{phone}")
def state(phone: str):
    s = SESSIONS.get(phone)
    return {"order": s["order"] if s else None,
            "orders": {k: v for k, v in ORDERS.items() if v["phone"] == phone}}
