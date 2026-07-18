import re
from collections import Counter
import pandas as pd
d = pd.read_parquet("/workspace/phrase-rl/data/contexts_train.parquet")
ins = d.instruction.astype(str).str.lower().str.strip()
print(f"contexts: {len(d)} | unique instruction strings: {ins.nunique()}")
STOP = {"the","a","an","to","of","and","on","in","from","into","onto","out","up","down",
        "with","at","right","left","side","it","its","then","them"}
VERBS = {"put","place","pick","move","take","open","close","fold","stack","turn","lift",
         "push","pull","set","grab","remove","slide","flip","wipe","sweep","pour","topple",
         "unfold","drape","lay","hang","insert","get","bring","drop","let","go","make"}
nouns = Counter()
pair_obj, pair_rec = Counter(), Counter()
for s in ins:
    toks = re.findall(r"[a-z]+", s)
    content = [t for t in toks if t not in STOP and t not in VERBS]
    nouns.update(content)
    m = re.match(r"(?:put|place|move|set|stack)\s+(?:the\s+)?(.+?)\s+(?:in|on|into|onto|to)\s+(?:the\s+)?(.+)", s)
    if m:
        pair_obj[m.group(1).strip()] += 1
        pair_rec[m.group(2).strip()] += 1
print(f"unique content words (obj/receptacle/attr vocab): {len(nouns)}")
print(f"unique OBJECT phrases (put-X-on-Y parse): {len(pair_obj)} | unique RECEPTACLES: {len(pair_rec)}")
print("top-15 content words:", nouns.most_common(15))
print("top-10 objects:", pair_obj.most_common(10))
print("top-10 receptacles:", pair_rec.most_common(10))
EVAL = ["carrot","spoon","eggplant","cube","block","coke","can","keyboard","wheel","tire",
        "ramekin","bowl","plate","towel","basket","sponge","pot","fridge","cloth"]
print("\neval-task noun coverage in the 2000 training contexts:")
for w in EVAL:
    print(f"  {w:10s} {nouns.get(w, 0)}")
