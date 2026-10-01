# smoke20.py <base-url> [out.json] | smoke20.py --diff <a.json> <b.json>
# A fixed 20-prompt set at temperature 0, thinking off, 96 tokens each. Saves answers to out.json; --diff counts
# prompts whose answers differ between two runs (a numerics change shows up here before it shows in GSM8K).
import json, sys, urllib.request, os, concurrent.futures as cf
P = ["What is the capital of Australia, and what is 17*23? Answer in one short sentence.",
     "Explain in two sentences why the sky is blue.", "Write a haiku about a GPU.",
     "List the first ten prime numbers.", "Translate 'The weather is nice today' into French, German and Spanish.",
     "What is the derivative of x^3 * sin(x)?", "Write a Python function that reverses a linked list.",
     "Summarize the plot of Romeo and Juliet in three sentences.", "What is 1234 * 5678?",
     "Name three differences between TCP and UDP.", "What year did the Apollo 11 mission land on the Moon, and who walked first?",
     "Give a JSON object with keys name, age, city for a fictional person.", "What is the boiling point of water at sea level in Kelvin?",
     "Explain what a hash table is to a ten year old.", "Write a SQL query that returns the top 5 customers by total order value.",
     "If a train travels 120 km in 1.5 hours, what is its average speed?", "What are the three states of matter? One line.",
     "Write the opening line of a mystery novel.", "Convert 0.375 to a fraction in lowest terms.",
     "What does the acronym GPU stand for, and what is it used for? Two sentences."]
if sys.argv[1] == "--diff":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    d = [i for i in range(len(P)) if a[i] != b[i]]
    print(f"smoke diff: {len(d)}/20 differ {d}")
    sys.exit(0)
base = sys.argv[1].rstrip("/")
def ask(p):
    body = {"model": os.environ.get("MODEL", "qwen3.8-27b"), "messages": [{"role": "user", "content": p}],
            "max_tokens": 96, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"].get("content") or ""
with cf.ThreadPoolExecutor(1) as ex: outs = list(ex.map(ask, P))
print(outs[0][:200].replace("\n", " "))
if len(sys.argv) > 2 and sys.argv[2]: json.dump(outs, open(sys.argv[2], "w"), indent=1)
