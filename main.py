import sys, time, torch, torch.nn.functional as F
from typing import Dict, Union
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ChoiceResult(BaseModel):
    choice: str; probabilities: Dict[str, float]; confidence: float
class NoulResult(BaseModel):
    noul: float; decision: bool
class SystemOneResponse(BaseModel):
    answers: Dict[str, Union[ChoiceResult, NoulResult]]; latency_ms: float

class ModernBERTSystemOne:
    def __init__(self, model_id, use_last_logit=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(self.device).eval()
        l2i = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.ei = -1 if use_last_logit else l2i["entailment"]

    def _logits(self, premise, hyps):
        x = self.tok([premise]*len(hyps), hyps, padding=True, truncation="only_first",
                     max_length=8192, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self.model(**x).logits.float()

    def evaluate_choice(self, state, criteria):
        lg = self._logits(state, [f"This issue is about {c.replace('_',' ')}." for c in criteria])
        probs = F.softmax(lg[:, self.ei], dim=0).tolist()
        pm = {c: round(p, 4) for c, p in zip(criteria, probs)}
        best = max(pm, key=pm.get)
        return ChoiceResult(choice=best, probabilities=pm, confidence=pm[best])

    def evaluate_noul(self, state, prop):
        p = F.softmax(self._logits(state, [prop])[0], dim=-1)[self.ei].item()
        return NoulResult(noul=round(p, 4), decision=p >= 0.5)

    def invoke(self, state, questions):
        t0 = time.perf_counter(); ans = {}
        for k, q in questions.items():
            if q["type"] == "choice": ans[k] = self.evaluate_choice(state, q["options"])
            elif q["type"] == "noul": ans[k] = self.evaluate_noul(state, q["instruction"])
            else: raise ValueError(f"unknown question type {q['type']!r}")
        return SystemOneResponse(answers=ans, latency_ms=round((time.perf_counter()-t0)*1000, 2))

schema = {
    "department": {"type": "choice", "options": ["billing", "technical_support", "sales_inquiry"]},
    "is_urgent": {"type": "noul", "instruction": "The customer needs help immediately."},
}
tickets = {
    "outage (yours)": "Customer states: 'The migration failed half-way through with code ERR_504. Our production database is locked and we are losing active checkout traffic. We need an incident engineer right now!'",
    "billing":        "Hi, I was charged twice for my March invoice. Could you refund the duplicate charge when you get a chance? Thanks.",
    "sales":          "We're a 200-person company evaluating your enterprise plan. Can someone send pricing for annual contracts and volume discounts?",
}
model_id = sys.argv[1] if len(sys.argv) > 1 else "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"; last = len(sys.argv) > 2
eng = ModernBERTSystemOne(model_id, use_last_logit=last)
eng.invoke(tickets["billing"], schema)  # warm-up
print(f"\n## {model_id}  ({'ORIGINAL logits[:, -1]' if last else 'FIXED entailment index'})")
for name, t in tickets.items():
    r = eng.invoke(t, schema); d, u = r.answers["department"], r.answers["is_urgent"]
    print(f"{name:15s} -> {d.choice:18s} conf={d.confidence:.3f}  urgent={str(u.decision):5s} P={u.noul:.3f}  {r.latency_ms:.0f}ms")
