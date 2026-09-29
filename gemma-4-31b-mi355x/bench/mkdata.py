# mkdata.py <tokenizer-dir> <gsm8k_test.jsonl> <out.jsonl>: natural-text prompts at the workload's shape
# (input uniform 1.9K..17.1K tokens, mean ~9.5K) built from GSM8K questions and worked answers, so MTP
# acceptance reflects real text instead of random-token continuations. Writes 1,200 prompts (about 33 MB).
# Run it where transformers is installed, e.g. inside the engine image (bench/README.md has the command).
import json, random, sys
from transformers import AutoTokenizer
tok_dir, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(tok_dir)
rows = [json.loads(l) for l in open(src)]
paras = [f"Problem: {r['question']}\nWorked solution: {r['answer']}\n" for r in rows]
lens = [len(tok(p, add_special_tokens=False).input_ids) for p in paras]
rnd = random.Random(7); out = []
for i in range(1200):
    target = rnd.randint(1900, 17100); idx = list(range(len(paras))); rnd.shuffle(idx)
    body, n = [], 0
    for j in idx:
        if n + lens[j] > target - 60: break
        body.append(paras[j]); n += lens[j]
    prompt = ("Below is a set of math problems with worked solutions.\n\n" + "".join(body) +
              "\nWrite a careful review of these solutions: for each of the first ten problems, restate the key step "
              "and check the arithmetic. Be thorough.")
    out.append({"prompt": prompt})
with open(dst, "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print("wrote", len(out))
