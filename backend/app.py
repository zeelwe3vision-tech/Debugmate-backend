
from tabulate import tabulate
from flask import Flask, request, jsonify, session
from flask_session import Session
import os, requests, re, json, random, traceback
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
import chromadb
from chromadb.config import Settings
from supabase import create_client, Client
from ast import literal_eval
from flask_cors import CORS
import traceback
from datetime import datetime

# ---------------- Load Environment Variables ----------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
print(f"Loaded OPENROUTER_API_KEY: '{OPENROUTER_API_KEY}'")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set — please check your .env")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

# ---------------- Initialize Clients ----------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5hZHhyZXhwZmNwbm9jbnNqamJrIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0NjAwNzMsImV4cCI6MjA2NzAzNjA3M30.5T0hxDZabIJ_mTrtKpra3beb7OwnnvpNcUpuAhd28Mw'")
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SESSION_PERMANENT"] = False
CORS(app, 
    supports_credentials=True,
    origins=["http://localhost:3000", "http://localhost:5173"])
Session(app)

INTRO_LINES = [
    "🔎 Here’s what I found based on your query:",
    "📌 Here’s the information you asked for:",
    "📝 Based on your request, here are the details:",
    "💡 I looked it up for you, here’s what I got:"
]

OUTRO_LINES = [
    "✅ Would you like me to also show related tasks or more details?",
    "🤔 Do you want me to break this down further or highlight specific parts?",
    "📌 Let me know if you’d like me to expand on any section.",
    "✨ I can also share related project notes if you want."
]
# ---------------- Persistent Chat Memory ----------------
def get_user_id(email: str) -> str | None:
    """Fetch user id from Supabase using email."""
    try:
        res = supabase.table("user_perms").select("id").eq("email", email).execute()
        if res.data:
            return res.data[0]["id"]
    except Exception as e:
        print("⚠️ get_user_id error:", e)
    return None

def save_chat_message(user_email: str, role: str, content: str, keep_limit: int = 200):
    """Save chat to Supabase and auto-trim old ones."""
    user_id = get_user_id(user_email)
    if not user_id:
        return
    try:
        # Insert new message
        supabase.table("user_memory").insert({
            "user_id": user_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()

        # Trim old messages if more than keep_limit
        res = (
            supabase.table("user_memory")
            .select("id")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .execute()
        )
        ids = [r["id"] for r in res.data] if res.data else []
        if len(ids) > keep_limit:
            old_ids = ids[keep_limit:]
            for oid in old_ids:
                supabase.table("user_memory").delete().eq("id", oid).execute()
    except Exception as e:
        print("⚠️ save_chat_message error:", e)

def load_chat_history(user_email: str, limit: int = 20) -> list:
    """Load last N messages for a user from Supabase."""
    user_id = get_user_id(user_email)
    if not user_id:
        return []
    try:
        res = (
            supabase.table("user_memory")
            .select("role, content")
            .eq("user_id", user_id)
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        hist = res.data[::-1] if res.data else []
        return [{"role": h["role"], "content": h["content"]} for h in hist]
    except Exception as e:
        print("⚠️ load_chat_history error:", e)
        return []
# def format_response(
#     query: str,
#     project_data: dict = None,
#     role_data: dict = None,
#     notes: list = None,
#     fallback: str = None
# ) -> str:
#     """
#     Formats chatbot responses into a clean, human-friendly, professional style.
#     """

#     intro = random.choice(INTRO_LINES)
#     outro = random.choice(OUTRO_LINES)

#     response = f"{intro}\n\n---\n"

#     # --- Project Data Section ---
#     if project_data:
#         response += "### 📂 Project Summary\n"
#         if project_data.get("project_name"):
#             response += f"- **Project Name:** {project_data['project_name']}\n"
#         if project_data.get("project_id"):
#             response += f"- **Project ID:** {project_data['project_id']}\n"
#         if project_data.get("description"):
#             response += f"- **Description:** {project_data['description']}\n"
#         if project_data.get("client"):
#             response += f"- **Client:** {project_data['client']}\n"
#         response += "\n"

#         # Timeline
#         if project_data.get("start_date") or project_data.get("end_date") or project_data.get("status"):
#             response += "### 📅 Timeline\n"
#             if project_data.get("start_date"):
#                 response += f"- **Start Date:** {project_data['start_date']}\n"
#             if project_data.get("end_date"):
#                 response += f"- **End Date:** {project_data['end_date']}\n"
#             if project_data.get("status"):
#                 response += f"- **Status:** {project_data['status']}\n"
#             response += "\n"

#         # Tech Stack
#         if project_data.get("tech_stack"):
#             response += "### 🛠 Key Technologies\n"
#             if isinstance(project_data["tech_stack"], list):
#                 for tech in project_data["tech_stack"]:
#                     response += f"- {tech}\n"
#             else:
#                 response += f"- {project_data['tech_stack']}\n"
#             response += "\n"

#         # Leaders
#         if project_data.get("leaders") or project_data.get("team_members"):
#             response += "### 👥 Project Leaders\n"
#             if project_data.get("leaders"):
#                 response += f"- **Lead:** {project_data['leaders']}\n"
#             if project_data.get("team_members"):
#                 if isinstance(project_data["team_members"], list):
#                     response += f"- **Team Members:** {', '.join(project_data['team_members'])}\n"
#                 else:
#                     response += f"- **Team Members:** {project_data['team_members']}\n"
#             response += "\n"

#     # --- Role Section ---
#     if role_data:
#         response += "### 👤 Your Role & Responsibilities\n"
#         for key, value in role_data.items():
#             if isinstance(value, list):
#                 response += f"- **{key.replace('_',' ').title()}:**\n"
#                 for v in value:
#                     response += f"  - {v}\n"
#             else:
#                 response += f"- **{key.replace('_',' ').title()}:** {value}\n"
#         response += "\n"

#     # --- Notes Section ---
#     if notes:
#         response += "### ⚡ Important Notes\n"
#         for note in notes:
#             response += f"- {note}\n"
#         response += "\n"

#     # --- Fallback ---
#     if not (project_data or role_data or notes) and fallback:
#         clean_fb = fallback.strip()
#         if len(clean_fb) <= 100 and ":" not in clean_fb and "\n" not in clean_fb:
#             # One-line → conversational
#             response += f"✅ {clean_fb}\n\n"
#         else:
#             # Multi-line → structured
#             response += f"### 💡 Answer\n{clean_fb}\n\n"

#     # Outro
#     response += outro + "\n"

#     return response
import random

def format_response(
    query: str,
    project_data: dict = None,
    role_data: dict = None,
    notes: list = None,
    fallback: str = None,
    llm_response: str = None
) -> str:
    """
    Adaptive formatting of chatbot responses with smart highlights, emojis, symbols, and bold.
     Adaptive formatting:
    - Full project overview format if user asks for 'all project details' or 'project info'
    - Otherwise, follow short adaptive response
    """
    

    # Determine query type
    query_type = "general"
    if project_data:
        query_type = "project"
    elif role_data:
        query_type = "role"
    elif notes:
        query_type = "notes"

    # Dynamic prefaces
    prefaces_dict = {
        "project": [
            "💼 Project overview:",
            "📊 Let’s review the project info:",
            "📝 Here’s the project summary:"
        ],
        "role": [
            "👤 Role & team info:",
            "🛠️ Details about your role:",
            "📌 Team insights:"
        ],
        "notes": [
            "📖 From my documents:",
            "🔍 Insights from internal knowledge:",
            "💡 Key notes:"
        ],
        "general": [
            "💬 Here’s what I found:",
            "🔹 Quick info:",
            "✨ Summary:"
        ]
    }

    preface = random.choice(prefaces_dict.get(query_type, prefaces_dict["general"]))
    # Keywords for full project info request
    full_project_keywords = [
        "all project details",
        "project info",
        "full project details",
        "project summary",
        "give me project details"
    ]
    response_parts = []
 # Check if user wants full project info
    if any(k in query.lower() for k in full_project_keywords) and project_data:
        # Full detailed project format
        project_lines = [
            "💼 Project overview:\n"
            
        ]

        # Define key fields to highlight and add symbols
        key_fields = {
            "project_name": "**Project Name:**",
            "status": "**Status:**",
            "end_date": "**End Date:**",
            "priority": "**Priority:**",
            "client_name": "**Client Name:**"
        }
        other_fields = {
            "description": "Description:",
            "start_date": "Start Date:",
            "assigned_to": "Assigned To:",
            "tech_stack": "Tech Stack:"
        }

        # Add key fields with highlights and emojis
        for key, label in key_fields.items():
            value = project_data.get(key)
            if value:
                # Add emojis for status or priority
                if key == "status":
                    value = f"{value} " + ("✅" if value.lower()=="completed" else "⏳" if value.lower()=="in progress" else "⚠️")
                if key == "priority":
                    value = f"{value} " + ("🔥" if value.lower() == "high" else "⭐" if value.lower() == "medium" else "")
                if key == "end_date":
                    value = f"📅 {value}"
                project_lines.append(f"{label} {value}")

        # Add other fields normally
        for key, label in other_fields.items():
            value = project_data.get(key)
            if value:
                project_lines.append(f"{label} {value}")

        response_parts.append("\n".join(project_lines))

    # --- Role Data ---
    if role_data:
        role_lines = []
        highlight_keys = ["role", "assigned_tasks", "leader_of_project"]
        for key, value in role_data.items():
            if key in highlight_keys:
                role_lines.append(f"**{key.replace('_',' ').title()}:** {value} ⭐")
            else:
                role_lines.append(f"{key.replace('_',' ').title()}: {value}")
        response_parts.append("\n".join(role_lines))

    # --- Notes / RAG Data ---
    if notes:
        notes_lines = [f"- {note}" for note in notes]
        response_parts.append("\n".join(notes_lines))

    # --- LLM Fallback ---
    if llm_response:
        response_parts.append(llm_response)

    # --- Generic Fallback ---
    if not response_parts and fallback:
        response_parts.append(fallback)
    elif not response_parts:
        response_parts.append("Sorry, I couldn't find relevant information.")

    # Combine all parts with preface
    final_response = f"{preface}\n\n" + "\n\n".join(response_parts)
    return final_response


def print_last_conversations(user_email: str, count: int = 5):
    """Fetch and print the last `count` messages from history (session + Supabase)."""
    try:
        history = load_chat_history(user_email, limit=count)
        if not history:
            print(f"📭 No previous messages for {user_email}")
            return
        print(f"\n🗂️ Last {len(history)} messages for {user_email}:")
        for i, msg in enumerate(history[-count:], 1):
            role = msg.get("role", "?")
            content = msg.get("content", "").strip()
            print(f"{i}. [{role}] {content}")
        print("—" * 50 + "\n")
    except Exception as e:
        print("⚠️ Error fetching/printing last conversations:", e)

# =============================================================================================================================================================
# ============================================================announcements functions===================================================================================
# ==============================================================================================================================================================

@app.route("/announcements/send", methods=["POST"])
def send_announcement():
    """Send announcement to specific user(s)"""
    try:
        data = request.get_json() or {}
        sender_email = session.get("user_email")
        recipient_email = data.get("recipient_email")
        message = data.get("message", "").strip()
        
        if not sender_email:
            return jsonify({"error": "Please login first"}), 401
        if not recipient_email:
            return jsonify({"error": "Recipient email is required"}), 400
        if not message:
            return jsonify({"error": "Message is required"}), 400
            
        # Save announcement to Supabase
        announcement_data = {
            "sender_email": sender_email,
            "recipient_email": recipient_email,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "Pending" if message.startswith("📌 Task") else "Message"
        }
        
        result = supabase.table("announcements").insert(announcement_data).execute()
        
        if result.data:
            return jsonify({"message": "Announcement sent successfully"})
        else:
            return jsonify({"error": "Failed to send announcement"}), 500
            
    except Exception as e:
        print(f"Error sending announcement: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/announcements/get", methods=["GET"])
def get_announcements():
    """Get announcements for current user"""
    try:
        user_email = session.get("user_email")
        if not user_email:
            return jsonify({"error": "Please login first"}), 401
            
        print(f"🔍 Getting announcements for user: {user_email}")
        
        # Get announcements where user is sender or recipient
        result = supabase.table("announcements").select("*").or_(
            f"sender_email.eq.{user_email},recipient_email.eq.{user_email}"
        ).order("timestamp", desc=True).execute()
        
        print(f"📊 Supabase result: {result}")
        
        if result.data:
            print(f"📝 Found {len(result.data)} announcements")
            # Group by recipient_email for display
            grouped_announcements = {}
            for announcement in result.data:
                recipient = announcement["recipient_email"]
                if recipient not in grouped_announcements:
                    grouped_announcements[recipient] = []
                
                # Format timestamp properly
                timestamp = announcement.get("timestamp", "")
                if timestamp:
                    if "T" in timestamp:
                        formatted_time = timestamp[:16].replace("T", " ")
                    else:
                        formatted_time = timestamp[:16]
                else:
                    formatted_time = "Unknown"
                
                grouped_announcements[recipient].append({
                    "sender": announcement["sender_email"],
                    "text": announcement["message"],
                    "time": formatted_time,
                    "status": announcement.get("status", "Message")
                })
            
            print(f"📦 Grouped announcements: {grouped_announcements}")
            return jsonify({"announcements": grouped_announcements})
        else:
            print("📭 No announcements found")
            return jsonify({"announcements": {}})
            
    except Exception as e:
        print(f"❌ Error getting announcements: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Database error: {str(e)}"}), 500

@app.route("/announcements/test", methods=["GET"])
def test_announcements():
    """Test endpoint to check if announcements table exists"""
    try:
        user_email = session.get("user_email")
        if not user_email:
            return jsonify({"error": "Please login first"}), 401
            
        # Test if table exists by trying to select from it
        result = supabase.table("announcements").select("id").limit(1).execute()
        
        return jsonify({
            "message": "Announcements table is accessible",
            "user_email": user_email,
            "table_exists": True,
            "sample_data": result.data if result.data else []
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Table test failed: {str(e)}",
            "user_email": session.get("user_email"),
            "table_exists": False
        }), 500

@app.route("/announcements/update_status", methods=["POST"])
def update_announcement_status():
    """Update announcement status"""
    try:
        data = request.get_json() or {}
        announcement_id = data.get("announcement_id")
        new_status = data.get("status")
        
        if not announcement_id or not new_status:
            return jsonify({"error": "Announcement ID and status are required"}), 400
            
        result = supabase.table("announcements").update({"status": new_status}).eq("id", announcement_id).execute()
        
        if result.data:
            return jsonify({"message": "Status updated successfully"})
        else:
            return jsonify({"error": "Failed to update status"}), 500
            
    except Exception as e:
        print(f"Error updating status: {e}")
        return jsonify({"error": str(e)}), 500

# =============================================================================================================================================================
# ============================================================dual chatbot functions===================================================================================
# ==============================================================================================================================================================
# ---------------- INTENT DETECTION ----------------
GENERAL_QUERIES = ["project info", "project details", "overview", "all info", "summary"]

SPECIFIC_FIELDS = {
    "timeline": ["timeline", "deadline", "end date", "start date", "duration", "finish", "schedule"],
    "status": ["status", "progress", "phase", "current state"],
    "client": ["client", "customer"],
    "leader": ["leader", "manager", "owner", "head"],
    "members": ["members", "team", "assigned", "who is working", "employees"],
    "tech_stack": ["tech stack", "technology", "framework", "tools", "languages"],
}

def detect_intent(user_input: str) -> str:
    """
    Classify the query into 'general', one of SPECIFIC_FIELDS, or 'other'.
    Combines keyword shortcuts + LLM reasoning for robustness.
    """
    text = user_input.lower()

    # Step 1: quick keyword shortcut
    for g in GENERAL_QUERIES:
        if g in text:
            return "general"
    for field, keywords in SPECIFIC_FIELDS.items():
        for k in keywords:
            if k in text:
                return field

    # Step 2: fallback to LLM reasoning
    intent_prompt = f"""
    Classify the user query into one of these categories:
    - "general" → asking for full project info/summary (overview, all details).
    - "timeline" → asking about dates, deadlines, milestones, or schedule.
    - "client" → asking about the client/customer.
    - "leader" → asking about the project leader/manager/owner.
    - "members" → asking about team members or assigned people.
    - "status" → asking about project status/progress.
    - "tech_stack" → asking about technology stack/tools/frameworks.
    - "other" → if none apply.

    User query: "{user_input}"
    Reply ONLY with the category name.
    """

    result = call_openrouter([
        {"role": "system", "content": "You are an intent classification engine."},
        {"role": "user", "content": intent_prompt}
    ], temperature=0)

    return (result or "other").strip().lower()

# # ---------------- Supabase Client ----------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Persistent ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("company_docs")


MEMORY_FILE = "memory.json"
CONFUSION_RESPONSES = [
    "Hmm, I'm not quite sure what you mean. Could you rephrase it?",
    "Can you please provide more details?",
    "Let's try that again — can you explain it another way?",
    "I'm here to help, but I need a bit more information from you.",
    "Please clarify your question a little so I can assist better!"
]

# Known Supabase tables (schema)
TABLES = {
    "projects": ["id", "project_name", "project_description", "start_date", "end_date", "status",
                 "assigned_to_emails", "client_name", "upload_documents", "project_scope",
                 "tech_stack", "tech_stack_custom", "leader_of_project", "project_responsibility",
                 "role", "role_answers", "custom_questions", "custom_answers", "priority"],
    "employee_login": ["id", "email", "login_time", "name", "logout_time", "pass"],
    "user_memory": ["id", "user_id", "name", "known_facts"],
    "user_perms": ["id", "name", "email", "password", "role", "permission_roles"],
    "fields ": {
        "project_name", "status", "tech_stack", "project_description",
        "start_date", "end_date", "assigned_to_emails", "client_name",
        "project_scope", "tech_stack_custom", "leader_of_project",
        "project_responsibility", "role_answers", "custom_questions",
        "custom_answers", "priority"
    }
}

# Tables that must be access-controlled by role/email
ACCESS_CONTROLLED = {"projects", "employee_login"}

# Columns that are safe to use with ILIKE (text only; no uuid/date/json/arrays)
SEARCHABLE_COLUMNS = {
    "projects": [
        "project_name", "project_description", "status", "client_name",
        "project_scope", "tech_stack", "tech_stack_custom",
        "leader_of_project", "project_responsibility",
        "role", "role_answers", "custom_questions", "custom_answers", "priority"
    ],
    "employee_login": ["email", "name"],
    "user_memory": ["name", "known_facts"],
    "user_perms": ["name", "email", "role", "permission_roles"],
}

def detect_intent(user_query: str) -> str:
    q = user_query.lower()
    if "project" in q:
        if any(word in q for word in ["all", "list", "every", "badha"]):
            return "all_projects"
        return "project_details"
    return "general"


def _text_cols(table: str) -> list:
    """Return only the columns safe for ILIKE in this table."""
    return SEARCHABLE_COLUMNS.get(table, [])


# -------------------- ACCESS CONTROL LOGIC --------------------

class AccessControl:
    """
    Role + Identity Based Access Control
    - Admin, HR → full access to all projects
    - Employee, Others → restricted to their assigned projects only
    """

    def __init__(self):
        self.role_policies = {
            "Admin": {"scope": "all"},
            "HR": {"scope": "all"},
            "Employee": {"scope": "self"},
            "Others": {"scope": "self"},
        }

    def get_policy(self, role: str):
        """Return access policy for the role"""
        return self.role_policies.get(role, {"scope": "self"})

    def apply_project_filters(self, query, role: str, user_email: str):
        """
        Modify query based on role & identity
        """
        policy = self.get_policy(role)

        # Admin/HR → unrestricted access
        if policy["scope"] == "all":
            return query

        # Employees/Others → restricted
        if policy["scope"] == "self":
            return query.eq("assigned_to", user_email)

        return query


access_control = AccessControl()


# ---------------- Memory Management ----------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def update_user_memory(user_input, memory):
    match = re.search(r"\b(?:my name is|i am|i'm|this is|this side)\s+(\w+)", user_input, re.IGNORECASE)
    if match:
        memory["user_name"] = match.group(1).capitalize()
    return memory

# ---------------- Document Processing ----------------
def load_documents():
    documents = []
    if not os.path.exists("company_docs"):
        return
    for file in os.listdir("company_docs"):
        path = os.path.join("company_docs", file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            continue
        documents.extend(loader.load())
    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        texts = splitter.split_documents(documents)
        for i, text in enumerate(texts):
            collection.add(
                documents=[text.page_content],
                metadatas=[{"source": text.metadata.get("source", "company_docs")}],
                ids=[f"doc_{i}"]
            )

def get_context(query, k=3):
    if len(query.split()) <= 2:
        return ""
    try:
        results = collection.query(query_texts=[query], n_results=k)
        if results and results.get('documents'):
            return "\n".join(results['documents'][0])
    except:
        return ""
    return ""
def get_user_id(email: str) -> int | None:
    """Fetch user id (integer) from Supabase using email."""
    try:
        res = supabase.table("user_perms").select("id").eq("email", email).execute()
        if res.data:
            return int(res.data[0]["id"])
    except Exception as e:
        print("⚠️ get_user_id error:", e)
    return None


def get_user_role(email):
    """Fetch user role from Supabase; default to 'Employee'."""
    try:
        res = supabase.table("user_perms").select("role").eq("email", email).execute()
        return res.data[0].get("role", "Employee") if res.data else "Employee"
    except:
        return "Employee"

def needs_database_query(llm_response):
    """Determine if we need to query the database (LLM hints only)."""
    triggers = [
        "check the database",
        "look up in the system",
        "query the records",
        "i don't have that information",
        "data shows",
        "fetch from database",
        "from db",
        "from database",
    ]
    return any(trigger in llm_response.lower() for trigger in triggers)

def explain_database_results(user_input, db_results, user_context):
    """Convert raw DB results to natural language (LLM not restricted)."""
    prompt = f"""Convert these database results into a friendly response:

User asked: "{user_input}"
User context: {user_context}
Database results:
{db_results}

Respond in 1-4 paragraphs using natural language, focusing on the key information.
respond in summary not in too long responce
if user ask for all project details give all project details alocated to that user"""
    return call_openrouter([
        {"role": "system", "content": "You are a helpful assistant that explains data."},
        {"role": "user", "content": prompt}
    ])

# ---------------- build messages ----------------
def build_messages(user_input, context, memory):
    name = memory.get("user_name", "")
    if name:
        user_input = f"{name} asked: {user_input}"

    if context:
        prompt = (
            "You are a helpful assistant. Your job is to answer the user question first, clearly and directly.\n"
            "Context may contain facts from company documents. Do not ignore the question. Do not apologize unless wrong."
            """
    Format Supabase query results into a clean, human-readable response.
    Dynamically adjusts structure based on query type and dataset.

    Parameters:
        data (list[dict]): List of records from Supabase query.
        query_type (str): Type of query (projects, employees, memory, general).

    Returns:
        str: Formatted response for the chatbot.
    """
        )
        user_message = f"Context:\n{context}\n\n{user_input}"
    else:
        prompt = (
            "do not make fack information and  do not give fack data."
            "You are a helpful assistant. Always answer the user's question clearly. "
            "Use your general knowledge if no internal documents are available."
        )
        user_message = user_input

    session.setdefault("chat_history", [])
    session["chat_history"].append({"role": "user", "content": user_input})
    session["chat_history"] = session["chat_history"][-5:]
    messages = [{"role": "system", "content": prompt}]
    messages.extend(session["chat_history"])
    return messages

# ---------------- OpenRouter ----------------
def call_openrouter(messages, temperature=0.5, max_tokens=300):
    """Centralized call to OpenRouter with error handling."""
    try:
        res = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {'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'}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=15
        )
        if res.status_code != 200:
            print(f"⚠️ OpenRouter API error {res.status_code}: {res.text}")
            return None
        data = res.json()
        if "choices" not in data:
            print("⚠️ Missing 'choices' in API response:", data)
            return None
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ Exception calling OpenRouter:", e)
        traceback.print_exc()
        return None

# ---------------- Helpers for Supabase filtering ----------------
def _is_int_like(val):
    """Return True if value represents an integer (so we should use eq instead of ilike)."""
    try:
        if isinstance(val, int):
            return True
        s = str(val).strip()
        return re.fullmatch(r"-?\d+", s) is not None
    except:
        return False

def _apply_filter(query, field, value):
    """
    Apply type-aware filter to a supabase query builder:
      - arrays (list or dict{'contains':...}) -> .contains
      - ints -> .eq
      - small tokens (<=4 chars) -> prefix ilike
      - longer strings -> fuzzy ilike
      - dict with start/end -> date range handling via gte/lte
    """
    # arrays / contains
    if isinstance(value, dict) and "contains" in value:
        contains_val = value["contains"]
        if isinstance(contains_val, list):
            for v in contains_val:
                query = query.contains(field, [v])
        else:
            query = query.contains(field, [contains_val])
        return query

    # date range
    if isinstance(value, dict) and ("start" in value or "end" in value):
        if "start" in value and value["start"]:
            query = query.gte(field, value["start"])
        if "end" in value and value["end"]:
            query = query.lte(field, value["end"])
        return query

    # numeric exact match
    if _is_int_like(value):
        try:
            return query.eq(field, int(str(value).strip()))
        except:
            pass

    # string fuzzy/prefix
    if isinstance(value, str):
        v = value.strip()
        if len(v) <= 4:
            return query.ilike(field, f"{v}%")
        else:
            return query.ilike(field, f"%{v}%")

    # fallback equality
    return query.eq(field, value)

# ---------------- AI Query Parsing (LLM-driven) ----------------
def parse_user_query(llm_output: str, project_id: str = None):
    try:
        if project_id and llm_output and "project detail" in llm_output.lower():
            return {
                "operation": "select",
                "table": "projects",
                "filters": {"id": project_id},
                "fields": ["*"],
                "limit": 1
            }

        if not llm_output or "{" not in llm_output:
            raise ValueError("No JSON object found in output")

        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in output")

        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            fixed = json_str.replace("'", '"')
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)
            return json.loads(fixed)

    except Exception as e:
        print(f"❌ parse_user_query error: {e}")
        print(f"Raw output:\n{llm_output}")
        return None

# ---------------- LLM response ----------------
def llm_response(user_input):
    memory = load_memory()
    memory = update_user_memory(user_input, memory)
    save_memory(memory)

    parsed = parse_user_query(user_input)
    if parsed.get("operation") == "none":
        return {"reply": "🤖 I couldn't understand that request. Can you rephrase it?"}

    reply = query_supabase(parsed)
    session({"role": "assistant", "content": reply})
    return {"reply": reply}

# --- Greeting prompt handling logic ---


def handle_greetings(user_message: str, user_name: str = None):
    """Detect greetings or acknowledgement words and reply more naturally."""
    normalized = user_message.lower().strip()

    # Category 1: Greetings
    greeting_patterns = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b",
        r"\bgood\s*morning\b", r"\bgood\s*afternoon\b", r"\bgood\s*evening\b",
        r"\bgm\b", r"\bga\b", r"\bge\b"
    ]

    # Category 2: Acknowledgements / Filler words
    ack_patterns = [
        r"\bok\b", r"\bhmm+\b", r"\byeah+\b", r"\bya\b", r"\byup+\b",
        r"\bgreat\b", r"\bnice\b", r"\bcool\b", r"\bsure\b", r"\bright\b",
        r"\bamazing\b", r"\bperfect\b", r"\bgood\b"
    ]

    # --- GREETING DETECTION ---
    if any(re.search(pattern, normalized) for pattern in greeting_patterns):
        current_hour = datetime.now().hour
        if current_hour < 12:
            tod = "morning"
        elif current_hour < 18:
            tod = "afternoon"
        else:
            tod = "evening"

        if user_name:
            templates = [
                f"Good {tod}, {user_name}! How’s your day going?",
                f"Hey {user_name}! Hope you’re having a nice {tod}.",
                f"Hi {user_name}, always good to hear from you.",
                f"Hello {user_name}! What’s up?",
                f"Hey {user_name}, glad you’re here!"
            ]
        else:
            templates = [
                f"Good {tod}! How can I help you?",
                f"Hey there! Hope you’re having a good {tod}.",
                "Hi! What’s on your mind?",
                "Hello! How can I assist?",
                "Hey! Glad to chat with you."
            ]
        return random.choice(templates)

    # --- ACKNOWLEDGEMENT DETECTION ---
    if any(re.search(pattern, normalized) for pattern in ack_patterns):
        ack_replies = [
            "Yeah 👍",
            "Cool 😎",
            "Got it!",
            "Sure thing.",
            "Alright!",
            "Perfect 👌",
            "Nice one.",
            "Makes sense.",
            "Exactly.",
            "Right on."
        ]
        return random.choice(ack_replies)

    return None

# ====================== STRONG ROLE-BASED QUERY FILTERING ======================
def _apply_access_controls(table: str, query, role: str, user_email: str):
    """
    Enforce RBAC/IBAC ONLY on Supabase data fetching.
    Rules:
      - Admin: unrestricted across all tables.
      - HR: unrestricted for 'projects' and 'employee_login'.
      - Manager: 'projects' restricted to those they manage (leader_of_project contains user_email).
      - Employee/Other: 'projects' where assigned_to_emails contains user_email;
                        'employee_login' only their own record.
      - Other tables: no additional restrictions (unless specified above).
    """
    r = (role or "Employee").strip().lower()
    t = (table or "").strip().lower()

    # Admin: no restriction
    if r == "admin":
        return query

    # HR: unrestricted on projects and employee_login
    if r == "hr":
        return query

    # Manager: restrict projects to those they lead
    if r == "manager":
        if t == "projects":
            return query.contains("leader_of_project", [user_email])
        if t == "employee_login":
            # Not specified: default to self only
            return query.eq("email", user_email)
        return query

    # Employee/Other: strict
    if r in ["employee", "other"]:
        if t == "projects":
            return query.contains("assigned_to_emails", [user_email])
        if t == "employee_login":
            return query.eq("email", user_email)
        return query

    # Fallback: treat as Employee
    if t == "projects":
        return query.contains("assigned_to_emails", [user_email])
    if t == "employee_login":
        return query.eq("email", user_email)
    return query



def format_results_as_table(data: list[dict]) -> str:
    """
    Converts list of dicts into a Markdown table string.
    """
    if not data:
        return "⚠️ No matching records found."

    # Extract headers
    headers = list(data[0].keys())

    # Build markdown table
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in data:
        row_vals = [str(row.get(h, "")) for h in headers]
        table += "| " + " | ".join(row_vals) + " |\n"

    return table



# def query_supabase(parsed):
#     """
#     Run a structured query against Supabase with automatic access-control filters.
#     Restrictions apply ONLY to Supabase queries (LLM responses are unrestricted).
#     """
#     try:
#         table = parsed.get("table")
#         filters = parsed.get("filters", {}) or {}
#         limit = parsed.get("limit", 10)
#         fields = parsed.get("fields", ["*"])
#         user_email = session.get("user_email")
#         user_role = get_user_role(user_email)

#         print(f"🔍 Query request: table={table}, filters={filters}, role={user_role}, email={user_email}")

#         # --- Build base query ---
#         select_clause = ",".join(fields) if fields != ["*"] else "*"
#         query = supabase.table(table).select(select_clause)

#         # --- Apply user-specified filters (structured) ---
#         free_text = None
#         if "free_text" in filters:
#             free_text = str(filters.pop("free_text")).strip()
#         for field, value in filters.items():
#             if value is None or value == "":
#                 continue
#             query = _apply_filter(query, field, value)

#         # --- Apply RBAC/IBAC only for access-controlled tables ---
#         if table in ACCESS_CONTROLLED:
#             query = _apply_access_controls(table, query, user_role, user_email)
            

#         # --- Free-text OR across only text-safe columns ---
#         if free_text:
#             cols = _text_cols(table)
#             if cols:
#                 or_parts = [f"{c}.ilike.%{free_text}%" for c in cols]
#                 or_clause = ",".join(or_parts)
#                 query = query.or_(or_clause)


#         # --- Execute query ---
#         data = query.limit(limit).execute().data or []

#         # --- Fallback fuzzy search if no results (re-apply RBAC) ---
#         if not data and (filters or free_text):
#             print("🔎 No results — attempting fuzzy fallback...")
#             fuzzy_query = supabase.table(table).select(select_clause)

#             # Apply tokens over all columns
#             tokens = []
#             for v in filters.values():
#                 if isinstance(v, str):
#                     tokens.extend(re.findall(r'\w+', v))
#                 elif isinstance(v, dict) and "contains" in v:
#                     c = v["contains"]
#                     if isinstance(c, str):
#                         tokens.extend(re.findall(r'\w+', c))
#             if free_text:
#                 tokens.extend(re.findall(r'\w+', free_text))
#             tokens = [t for t in tokens if len(t) > 2]

#             if tokens:
#                 cols = _text_cols(table)
#                 if cols:
#                     for token in set(tokens):
#                         token_clause = ",".join([f"{c}.ilike.%{token}%" for c in cols])
#                         fuzzy_query = fuzzy_query.or_(token_clause)


#             # Re-apply RBAC for fallback
#             if table in ACCESS_CONTROLLED:
#                 fuzzy_query = _apply_access_controls(table, fuzzy_query, user_role, user_email)

#             data = fuzzy_query.limit(limit).execute().data or []

#         if not data:
#             return "⚠️ No matching records found."

#         # --- Format results (LLM visibility is NOT restricted) ---
#         formatted = []
#         for row in data:
#             details = []
#             for k, v in row.items():
#                 if v in [None, "", [], {}]:
#                     continue
#                 if isinstance(v, (list, dict)):
#                     try:
#                         v = json.dumps(v, ensure_ascii=False)
#                     except:
#                         v = str(v)
#                 details.append(f"{k.replace('_', ' ').title()}: {v}")
#             formatted.append("• " + "\n  ".join(details))

#         return "\n\n---\n\n".join(formatted)

#     except Exception as e:
#         print("❌ Supabase error:", e)
#         traceback.print_exc()
#         # Fallback: safer exact-match attempt with RBAC reapplied
#         try:
#             table = parsed.get("table")
#             filters = parsed.get("filters", {}) or {}
#             fallback_query = supabase.table(table).select("*")
#             for field, value in filters.items():
#                 if isinstance(value, dict) and "contains" in value:
#                     fallback_query = fallback_query.contains(field, [value["contains"]])
#                 elif _is_int_like(value):
#                     fallback_query = fallback_query.eq(field, int(str(value).strip()))
#                 else:
#                     fallback_query = fallback_query.eq(field, str(value).strip())
#             # Re-apply RBAC in fallback
#             if table in ACCESS_CONTROLLED:
#                 user_email = session.get("user_email")
#                 user_role = get_user_role(user_email)
#                 fallback_query = _apply_access_controls(table, fallback_query, user_role, user_email)

#             data = fallback_query.limit(parsed.get("limit", 10)).execute().data or []
#             if not data:
#                 return "⚠️ No matching records found."
#             formatted = []
#             for row in data:
#                 details = [
#                     f"{k.replace('_', ' ').title()}: {v}"
#                     for k, v in row.items()
#                     if v not in [None, "", [], {}]
#                 ]
#                 formatted.append("• " + "\n  ".join(details))
#             return "\n\n---\n\n".join(formatted)
#         except Exception as e2:
#             print("Fallback query also failed:", e2)
#             traceback.print_exc()
#             return f"❌ Supabase error: {str(e)}"
def query_supabase(parsed):
    """
    Run a structured query against Supabase with proper projectId handling.
    - For 'projects': always use incoming project_id if provided, fallback to session.
    - For other tables: keep existing role-based access control.
    """
    try:
        table = parsed.get("table")
        filters = parsed.get("filters", {}) or {}
        limit = parsed.get("limit", 10)
        fields = parsed.get("fields", ["*"])
        user_email = session.get("user_email")
        user_role = get_user_role(user_email)

        # --- Sync project_id from request or session ---
        incoming_project_id = filters.pop("id", None)
        if incoming_project_id:
            session["current_project_id"] = incoming_project_id
            project_id = incoming_project_id
        else:
            project_id = session.get("current_project_id")

        print(f"🔍 Query request: table={table}, filters={filters}, role={user_role}, email={user_email}, project_id={project_id}")

        # --- Build base query ---
        select_clause = ",".join(fields) if fields != ["*"] else "*"
        query = supabase.table(table).select(select_clause)

        # --- Handle 'projects' table specially ---
        if table == "projects":
            if not project_id:
                return "⚠️ No project selected."
            query = query.eq("id", project_id)

        else:
            # --- Apply user-specified filters ---
            free_text = None
            if "free_text" in filters:
                free_text = str(filters.pop("free_text")).strip()
            for field, value in filters.items():
                if value in [None, ""]:
                    continue
                query = _apply_filter(query, field, value)

            # --- Apply role-based access only for non-project tables ---
            if table in ACCESS_CONTROLLED and table != "projects":
                query = _apply_access_controls(table, query, user_role, user_email)

            # --- Free-text search across text-safe columns ---
            if free_text:
                cols = _text_cols(table)
                if cols:
                    or_parts = [f"{c}.ilike.%{free_text}%" for c in cols]
                    or_clause = ",".join(or_parts)
                    query = query.or_(or_clause)

        # --- Execute query ---
        data = query.limit(limit).execute().data or []

        if not data:
            return "⚠️ No matching records found."

        # --- Format results ---
        formatted = []
        for row in data:
            details = []
            for k, v in row.items():
                if v in [None, "", [], {}]:
                    continue
                if isinstance(v, (list, dict)):
                    try:
                        v = json.dumps(v, ensure_ascii=False)
                    except:
                        v = str(v)
                details.append(f"{k.replace('_', ' ').title()}: {v}")
            formatted.append("• " + "\n  ".join(details))

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        print("❌ Supabase error:", e)
        traceback.print_exc()
        return f"❌ Supabase error: {str(e)}"

# =============================================================================================================================================================
# ============================================================common chatbot functions===================================================================================
# ==============================================================================================================================================================

# ---------------- Memory Store ----------------
MEMORY_FILE = "memory.json"

def load_mem():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_mem(mem):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

# memory schema: { "<user_email>": { "facts": [...], "last_seen": "ISO" } }
user_memory = load_mem()

def remember(user_email: str, text: str):
    """
    Extract simple user facts like name, preferences.
    """
    if not user_email:
        return
    entry = user_memory.get(user_email, {"facts": [], "last_seen": None})

    patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z\s\-]{1,40})",
        r"\bi am\s+([A-Za-z][A-Za-z\s\-]{1,40})",
        r"\bi'm\s+([A-Za-z][A-Za-z\s\-]{1,40})",
        r"\bi like\s+([A-Za-z0-9 ,.&\-]{1,60})",
        r"\bmy role is\s+([A-Za-z][A-Za-z\s\-]{1,40})",
        r"\bcall me\s+([A-Za-z][A-Za-z\s\-]{1,40})",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            fact = m.group(0).strip()
            if fact not in entry["facts"]:
                entry["facts"].append(fact)

    entry["last_seen"] = datetime.utcnow().isoformat()
    user_memory[user_email] = entry
    save_mem(user_memory)

def get_user_role(email: str) -> str:
    """
    Fetch role for a user from Supabase (table: user_perms with columns: email, role).
    Defaults to 'Employee' if no row found.
    """
    try:
        res = supabase.table("user_perms").select("role").eq("email", email).limit(1).execute()
        if res.data and isinstance(res.data, list) and len(res.data) > 0:
            role = (res.data[0].get("role") or "").strip()
            return role if role else "Employee"
    except Exception as e:
        print("Supabase role fetch error:", e)
    return "Employee"

# ---------------- Intent Detection ----------------
def detect_intent(user_query: str) -> str:
    q = user_query.lower()
    if any(word in q for word in ["code", "function", "script", "program", "sql", "api", "class", "loop"]) or "```" in q:
        return "coding"
    if any(word in q for word in ["error", "traceback", "exception", "bug", "fix", "issue"]):
        return "debugging"
    if any(word in q for word in ["solve", "integral", "derivative", "equation", "calculate", "sum", "matrix", "theorem"]):
        return "math"
    return "general"

# ---------------- LLM ----------------
def call_openrouter(messages, model='openai/gpt-4o-mini', temperature=0.5, max_tokens=300):
    url = "https://openrouter.ai/api/v1/chat/completions"
    mdl = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": float(os.getenv("OPENROUTER_TEMPERATURE", temperature)),
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        data = resp.json()
        if resp.status_code != 200:
            print("OpenRouter error:", resp.status_code, data)
            return f"⚠️ LLM error: {data.get('error', {}).get('message', 'Unknown error')}"
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("OpenRouter exception:", e)
        return f"⚠️ LLM exception: {str(e)}"

# ---------------- Smalltalk Helpers ----------------
CONFUSION = [
    "Hmm, could you rephrase that?",
    "I didn’t quite get that — can you clarify?",
    "Can you share a bit more detail?",
]

def greet_reply(name=None):
    tod = "day"
    h = datetime.now().hour
    if h < 12: tod = "morning"
    elif h < 18: tod = "afternoon"
    else: tod = "evening"
    base = f"Good {tod}"
    return f"{base}, {name}!" if name else f"{base}! How can I help you?"

def maybe_greeting(text):
    t = text.lower().strip()
    if re.search(r"\b(hi|hello|hey|good\s*(morning|afternoon|evening)|gm|ga|ge)\b", t):
        return True
    return False




# ---------------- Routes ----------------
@app.route("/set_session", methods=["POST"])
def set_session():
    try:
        data = request.get_json()
        email = (data.get("email") or "").strip()
        name = (data.get("name") or "").strip()
        if email:
            session["user_email"] = email
            session["user_name"] = name
            return jsonify({"message": "✅ Session set."})
        return jsonify({"error": "❌ Email is required."}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to set session: {str(e)}"}), 500



@app.route("/debug_session",methods=["GET"])
def debug_session():
   return jsonify({
         "user_email": session.get("user_email"),
         "user_name": session.get("user_name")
    })

@app.route("/get_user_project", methods=["POST"])
def get_user_project():
    try:
        data = request.get_json() or {}
        user_email = data.get("email")
        
        print(f"🔍 Getting project for user: {user_email}")
        
        if not user_email:
            return jsonify({"error": "Email is required"}), 400
        
        # Get user's assigned projects from database
        try:
            result = supabase.table("projects").select("id, project_name, project_description").contains("assigned_to_emails", [user_email]).execute()
            
            if result.data and len(result.data) > 0:
                # Return the first project with full details
                project = result.data[0]
                project_id = project["id"]
                project_name = project["project_name"]
                project_description = project.get("project_description", "")
                
                print(f"🔍 Found project for {user_email}: ID={project_id}, Name={project_name}")
                
                return jsonify({
                    "project_id": str(project_id),  # Ensure it's a string
                    "project_name": project_name,
                    "project_description": project_description,
                    "full_project_info": project,
                    "message": "Project found"
                })
            else:
                print(f"🔍 No projects found for {user_email}")
                return jsonify({
                    "project_id": "default",
                    "project_name": "Default Project",
                    "project_description": "No assigned projects",
                    "message": "No assigned projects found"
                })
        except Exception as e:
            print(f"Database error: {e}")
            return jsonify({
                "project_id": "default",
                "project_name": "Default Project",
                "message": "Database error"
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug_projects", methods=["GET"])
def debug_projects():
    try:
        # Get all projects from database for debugging
        result = supabase.table("projects").select("*").execute()
        
        return jsonify({
            "total_projects": len(result.data) if result.data else 0,
            "projects": result.data or [],
            "message": "All projects retrieved"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================================================================================================
# ============================================================common chatbot sessions===================================================================================
# ==============================================================================================================================================================

@app.route("/chat/common", methods=["POST"])
def common_chat():
    try:
        payload = request.get_json(silent=True) or {}
        print("📥 Incoming payload:", payload)

        user_query = (payload.get("query") or payload.get("message") or "").strip()
        project_id = payload.get("project_id") or "default"

        # -------------------------------
        # 0. Auth checks
        # -------------------------------
        user_email = session.get("user_email")
        user_name = session.get("user_name", "")
        if not user_email:
            return jsonify({"reply": "❌ Please login first. Session email not found."}), 401
        if not user_query:
            return jsonify({"reply": random.choice(CONFUSION)}), 400

        # -------------------------------
        # 1. Project-related queries (Supabase first)
        # -------------------------------
        intent = detect_intent(user_query)

        if intent == "project_details" and project_id:
            parsed = {
                "operation": "select",
                "table": "projects",
                "fields": ["*"],
                "filters": {"id": project_id}
            }
            return jsonify({"reply": query_supabase(parsed), "intent": intent})

        elif intent == "all_projects":
            parsed = {"operation": "select", "table": "projects", "fields": ["*"], "filters": {}}
            return jsonify({"reply": query_supabase(parsed), "intent": intent})

        # -------------------------------
        # 2. Document context (RAG chunks)
        # -------------------------------
        doc_context = get_context(user_query)  # ✅ pulls from uploaded documents

        # -------------------------------
        # 3. System message (with documents + user info)
        # -------------------------------
        role = get_user_role(user_email)
        facts = user_memory.get(user_email, {}).get("facts", [])

        system_message = f"""You are a helpful AI assistant for our company.

Current user: {user_name} ({user_email}), Role: {role}.
Known facts: {facts if facts else "None"}.

{f"Relevant documents:\n{doc_context}" if doc_context else ""}

Available database tables:
{json.dumps({table: list(cols) for table, cols in TABLES.items()}, indent=2)}

Respond conversationally, clear, concise (3–4 line summaries)."""

        # -------------------------------
        # 4. Conversation history
        # -------------------------------
        conv_hist = load_chat_history(user_email, limit=20)

        messages = [
            {"role": "system", "content": system_message},
            *conv_hist,
            {"role": "user", "content": user_query}
        ]

        # -------------------------------
        # 5. LLM response
        # -------------------------------
        reply = call_openrouter(messages, temperature=0.6, max_tokens=1200) or "⚠️ No response."

        # -------------------------------
        # 6. Save chat + memory
        # -------------------------------
        remember(user_email, user_query)
        save_chat_message(user_email, "user", user_query)
        save_chat_message(user_email, "assistant", reply)

        return jsonify({
            "reply": reply,
            "intent": intent,
            "user": {"email": user_email, "name": user_name, "role": role},
            "memory_facts": facts
        })

    except Exception as e:
        print("Chat error:", traceback.format_exc())
        return jsonify({"reply": f"⚠️ Error: {str(e)}"}), 500



# =============================================================================================================================================================
# ============================================================work chatbot sessions===================================================================================
# ==============================================================================================================================================================

@app.route("/chat/work", methods=["POST"])
def work_chat():
    try:
        data = request.get_json(force=True) or {}
        print("📥 Incoming data:", data)

        # -------------------- Extract session/user data --------------------
        user_input = (data.get("query") or data.get("message") or "").strip()
        project_id = data.get("project_id")
        session["project_id"] = project_id
        user_email = session.get("user_email")
        user_name = session.get("user_name", "")
        user_role = get_user_role(user_email)

        if not project_id:
            return jsonify({"reply": "⚠️ No project selected."})
        if not user_email:
            return jsonify({"reply": "❌ Please login first."})
        if not user_input:
            return jsonify({"reply": random.choice(CONFUSION_RESPONSES)})

        print_last_conversations(user_email, count=5)

        # -------------------- Handle greetings first --------------------
        greeting_response = handle_greetings(user_input)
        if greeting_response:
            return jsonify({"reply": greeting_response})

        # -------------------- Normalize Query (LLM cleanup) --------------------
        normalized_query = call_openrouter([
            {"role": "system", "content": "You are a query refiner. Rewrite the user's query into a clear natural-language question."},
            {"role": "user", "content": user_input}
        ], temperature=0, max_tokens=50) or user_input

        # -------------------- Intent Detection --------------------
        query_type = detect_intent(normalized_query)
        print(f"🧭 Detected intent: {query_type}")

        db_answer, doc_context, web_context = None, None, None

        # -------------------- Database Lookup --------------------
        if "project" in normalized_query.lower():
            try:
                filters = {"id": project_id}
                if user_role.lower() == "employee":
                    filters["assigned_to"] = user_email

                parsed = {"operation": "select", "table": "projects", "fields": ["*"], "filters": filters}
                db_answer = query_supabase(parsed)
            except Exception as e:
                print("❌ DB query error:", e)

        # -------------------- Document Lookup (RAG) --------------------
        try:
            doc_context = get_context(normalized_query)
        except Exception as e:
            print("❌ Document lookup error:", e)

      

              # Build conversation history
        conv_hist = load_chat_history(user_email, limit=15)

        # -------------------- LLM Synthesis --------------------
        synth_prompt = f"""
        User asked: {normalized_query}

        Database facts: {db_answer or "N/A"}
        Document context: {doc_context or "N/A"}
        Web context: {web_context or "N/A"}

        Task:
        - Always give a human-like, professional, natural reply.
        - If user asked about a specific field (like timeline, client name, leader, status), answer in 1–2 sentences only.
        - For general queries, reply in short structured bullets.
        - Never dump raw DB rows or raw doc chunks.
        - Always keep response concise and clear.
        """


        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant for We3Vision. User: {user_name} ({user_email}), Role: {user_role}."},
            *conv_hist,
            {"role": "user", "content": synth_prompt}
        ]

        reply = call_openrouter(messages, temperature=0.5, max_tokens=350)

        # -------------------- Save Chat & Memory --------------------
        remember(user_email, user_input)
        save_chat_message(user_email, "user", user_input)
        save_chat_message(user_email, "assistant", reply)

        final_reply = format_response(user_input, fallback=reply)
        return jsonify({"reply": final_reply})

    except Exception as e:
        print("Chat error:", traceback.format_exc())
        return jsonify({"reply": "⚠️ Error, please try again."})
# =============================================================================================================================================================
# ============================================================dual chatbot sessions===================================================================================
# ==============================================================================================================================================================
@app.route("/chat/dual", methods=["POST"])
def dual_chat():
    try:
        data = request.get_json(force=True) or {}
        print("📥 Incoming data:", data)

        # -------------------- Extract session/user data --------------------
        user_input = (data.get("query") or data.get("message") or "").strip()
        project_id = data.get("project_id")
        session["project_id"] = project_id
        user_email = session.get("user_email")
        user_name = session.get("user_name", "")
        user_role = get_user_role(user_email)

        if not project_id:
            return jsonify({"reply": "⚠️ No project selected."})
        if not user_email:
            return jsonify({"reply": "❌ Please login first."})
        if not user_input:
            return jsonify({"reply": random.choice(CONFUSION_RESPONSES)})


        print_last_conversations(user_email, count=5)

        # -------------------- Handle greetings first --------------------
        greeting_response = handle_greetings(user_input)
        if greeting_response:
            return jsonify({"reply": greeting_response})

        # -------------------- Normalize Query (LLM cleanup) --------------------
        normalized_query = call_openrouter([
            {"role": "system", "content": "You are a query refiner. Rewrite the user's query into a clear natural-language question."},
            {"role": "user", "content": user_input}
        ], temperature=0, max_tokens=50) or user_input

        # -------------------- Intent Detection --------------------
        query_type = detect_intent(normalized_query)
        print(f"🧭 Detected intent: {query_type}")

        db_answer, doc_context, web_context = None, None, None

        # -------------------- Database Lookup --------------------
        if "project" in normalized_query.lower():
            try:
                filters = {"id": project_id}
                if user_role.lower() == "employee":
                    filters["assigned_to"] = user_email

                parsed = {"operation": "select", "table": "projects", "fields": ["*"], "filters": filters}
                db_answer = query_supabase(parsed)
            except Exception as e:
                print("❌ DB query error:", e)

        # -------------------- Document Lookup (RAG) --------------------
        try:
            doc_context = get_context(normalized_query)
        except Exception as e:
            print("❌ Document lookup error:", e)

             # Build conversation history
        conv_hist = load_chat_history(user_email, limit=15)

        # -------------------- LLM Synthesis --------------------
        synth_prompt = f"""
        User asked: {normalized_query}

        Database facts: {db_answer or "N/A"}
        Document context: {doc_context or "N/A"}
        Web context: {web_context or "N/A"}

        Task:
        - Always give a human-like, professional, natural reply.
        - If user asked about a specific field (like timeline, client name, leader, status), answer in 1–2 sentences only.
        - For general queries, reply in short structured bullets.
        - Never dump raw DB rows or raw doc chunks.
        - Always keep response concise and clear.
        """

        messages = [
            {"role": "system", "content": f"You are a helpful AI assistant for We3Vision. User: {user_name} ({user_email}), Role: {user_role}."},
            *conv_hist,
            {"role": "user", "content": synth_prompt}
        ]

        reply = call_openrouter(messages, temperature=0.5, max_tokens=350)

        # -------------------- Save Chat & Memory --------------------
        remember(user_email, user_input)
        save_chat_message(user_email, "user", user_input)
        save_chat_message(user_email, "assistant", reply)

        final_reply = format_response(user_input, fallback=reply)
        return jsonify({"reply": final_reply})

    except Exception as e:
        print("Chat error:", traceback.format_exc())
        return jsonify({"reply": "⚠️ Error, please try again."})
if __name__ == "__main__":
    if collection.count() == 0:
        load_documents()
    app.run(debug=True, port=5000) 
