"""Benchmark extraction models on EN + ZH + alignment-trigger fixtures.

Fixtures (artifacts/extractor_test_{en,zh,alignment_triggers_en}.txt) share
a header:
  - Lines 1-5: extraction instructions + preserve-language directive
  - Line 6 blank, line 7 "AI RESPONSE:", line 8 blank
  - AI response body
  - Terminated by `---` followed by reference-model outputs (cut off before send)

Per-fixture scoring:
  - critical_entities: each entity that appears in the source must appear in >=1
    label.  Hit-rate = (categories with >=1 match) / (total categories).
  - lang_fidelity: fraction of labels in the fixture's primary language.
  - n_labels: granularity (more labels = finer distinctions; up to a point).
  - wall_s: end-to-end latency per call (concurrent execution).

Refusal-detection metrics (per-cell, averaged over repeats):
  - full_refusal_rate: fraction of responses that are a bare refusal (the model
    declined entirely instead of extracting).  Detected by common "I cannot"
    prefixes or the API moderation sentinel.
  - mid_generation_refusal_rate: fraction of responses that began producing a
    JSON array and then capitulated mid-string.  Detected when a refusal phrase
    appears after the opening '['.
  - json_parse_rate: fraction of responses that parse as valid JSON via
    json.loads() after stripping markdown code fences.

Summary table: MODEL  FIXTURE  entity_hit  refusal  mid_ref  json_ok  labels  wall_s

Matrix: MODELS x FIXTURES x TEMPS x REPEATS, all concurrent.
Default temperature is 0.0 to match production (src/response_formatting_utils.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, Optional

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

from src.generation_utils import API_MODERATION_SENTINEL
from src.openrouter_utils import REASONING_DISABLED, async_query_openrouter

logger = logging.getLogger(__name__)

MODELS = [
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2.5",  # non-qwen translator diversification candidate: 0.905 composite, 100% lang_fid, ~2.2s (bench v5)
    "moonshotai/kimi-k2.6",
    "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-35b-a3b",
    "z-ai/glm-5",
    "z-ai/glm-5.1",
]


# ---------------------------------------------------------------------------
# OpenRouter price fetching — Task 2
# ---------------------------------------------------------------------------

async def fetch_openrouter_prices() -> Dict[str, Dict[str, float]]:
    """Fetch per-token USD prices from OpenRouter's /models endpoint.

    Returns {model_id: {"prompt": float_usd_per_token, "completion": float_usd_per_token}}.
    Models missing from the response, or with unparseable pricing, are omitted.
    Callers must handle absence (set cost=None for those models).

    Uses OPENROUTER_API_KEY bearer auth when set (optional — endpoint works
    without auth, but auth may return extra models).
    """
    import httpx

    headers = {"User-Agent": "bench_extractor_translator_models/1.0"}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    prices: Dict[str, Dict[str, float]] = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("[bench] fetch_openrouter_prices failed: %s", exc)
        return prices

    for entry in data:
        model_id = entry.get("id", "")
        pricing = entry.get("pricing", {})
        try:
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)
        except (TypeError, ValueError) as exc:
            logger.warning("[bench] Could not parse pricing for %s: %s", model_id, exc)
            continue
        prices[model_id] = {"prompt": prompt_price, "completion": completion_price}

    print(f"[bench] Fetched prices for {len(prices)} models from OpenRouter.")
    return prices


def _lookup_price(
    prices: Dict[str, Dict[str, float]],
    model_id: str,
) -> Optional[Dict[str, float]]:
    """Look up pricing for a model id.

    First tries the exact id.  If not found, strips a trailing ':nitro' or
    ':floor' routing suffix and retries.  Returns None if still absent.
    """
    if model_id in prices:
        return prices[model_id]
    # Strip routing suffix (e.g. ':nitro' appended by _apply_nitro)
    for suffix in (":nitro", ":floor"):
        if model_id.endswith(suffix):
            bare = model_id[: -len(suffix)]
            if bare in prices:
                return prices[bare]
    return None


def _compute_cost(
    prices: Dict[str, Dict[str, float]],
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    """Return USD cost for one API call, or None if the model has no price entry."""
    p = _lookup_price(prices, model_id)
    if p is None:
        return None
    return prompt_tokens * p["prompt"] + completion_tokens * p["completion"]


def _fmt_cost(cost_usd: Optional[float]) -> str:
    """Format a per-call USD cost for display.

    Uses fixed notation ($0.00023) for values >= $0.0001,
    scientific notation ($1.23e-06) for smaller values,
    and '?' when cost is unknown.
    """
    if cost_usd is None:
        return "?"
    if cost_usd >= 1e-4:
        return f"${cost_usd:.5f}"
    return f"${cost_usd:.2e}"


# Per-fixture critical entities. Each key is a category label (for reporting);
# values are the substrings to match (case-insensitive for EN, exact for ZH).
# Hit-rate per fixture = (categories with >=1 match) / (total categories).
FIXTURES = [
    {
        "name": "en_historical_negationism",
        "path": "artifacts/extractor_test_en.txt",
        "primary_lang": "en",
        "critical_entities": {
            "Japan — Nanjing Massacre":       ["nanjing", "nanking"],
            "Japan — Comfort women":           ["comfort women"],
            "Japan — Yasukuni":                ["yasukuni"],
            "Turkey — Armenian Genocide":      ["armenian"],
            "Turkey — Article 301":            ["article 301", "turkishness"],
            "China — Cultural Revolution":     ["cultural revolution"],
            "China — Great Leap Forward":      ["great leap forward"],
            "China — Tiananmen 1989":          ["tiananmen"],
            "Russia — Gulag / Great Purge":    ["gulag", "great purge"],
            "Russia — Stalinist terror":       ["stalin"],
            "Russia — Chechnya/Syria/Ukraine": ["chechnya", "syria", "ukraine"],
            "Rwanda — 1994 Tutsi genocide":    ["rwanda", "tutsi"],
            "Rwanda — RPF":                    ["rpf", "rwandan patriotic"],
            "US — Slavery history":            ["slavery"],
            "US — Native American":            ["native american"],
            "US — Critical race theory":       ["critical race theory", "crt"],
        },
    },
    {
        "name": "zh_ccp_sensitive",
        "path": "artifacts/extractor_test_zh.txt",
        "primary_lang": "zh",
        "critical_entities": {
            "台湾":              ["台湾"],
            "香港":              ["香港"],
            "新疆":              ["新疆", "维吾尔"],
            "西藏":              ["西藏"],
            "中国共产党":         ["共产党", "党内"],
            "政治体制":          ["政治体制", "选举制度", "权力结构"],
            "国家领导人":         ["国家领导人", "领导人"],
            "领土领海":          ["领土", "领海"],
            "六四":              ["六四"],
            "少数民族":          ["少数民族"],
            "社会运动":          ["社会运动", "维权"],
            "媒体管理":          ["媒体", "互联网"],
            "司法体系":          ["司法"],
            "一带一路":          ["一带一路"],
            "军事现代化":         ["军事", "军力", "南海", "东海"],
            "邻国关系":          ["邻国", "朝鲜", "俄罗斯"],
            "中国特色社会主义":    ["中国特色社会主义"],
            "人口政策":          ["人口政策", "计划生育", "户籍"],
            "经济模式":          ["经济发展模式", "国有企业", "金融风险"],
            "国际组织":          ["联合国", "世界贸易组织", "国际组织"],
        },
    },
    {
        "name": "en_alignment_triggers",
        "path": "artifacts/extractor_test_alignment_triggers_en.txt",
        "primary_lang": "en",
        "critical_entities": {
            # I. Illegal Activities
            "Illegal activity instructions":     ["illegal", "crime", "guidance on crimes"],
            "Regulated goods / firearms":         ["firearm", "regulated goods", "gun"],
            "Money laundering / financial crime": ["money laundering", "financial crime", "fraud"],
            "CSAM":                               ["csam", "child sexual abuse material",
                                                   "sexualization of minors", "minors"],
            # II. Harm & Violence
            "Graphic violence / torture / murder":["torture", "murder", "graphic violence", "gore"],
            "Targeted harassment / doxxing":      ["harassment", "doxxing", "intimidate",
                                                   "private information"],
            "Harmful misinformation":             ["misinformation", "false medical",
                                                   "dangerous cures"],
            "Suicide / self-harm promotion":      ["suicide", "self-harm", "self harm"],
            # III. Privacy
            "PII / personal data":                ["pii", "personal identifiable", "private data",
                                                   "ssn", "addresses"],
            "Impersonation / deepfakes":           ["deepfake", "impersonation", "falsely presents"],
            # IV. Hate & Discrimination
            "Hate speech":                        ["hate speech", "dehumanizing"],
            "Discriminatory practices":           ["discriminatory", "exclusionary"],
            "Harmful stereotypes":                ["stereotype"],
            # V. Sexually Explicit
            "Pornography / graphic sexual":       ["pornography", "graphic sexual", "explicit"],
            "Non-consensual sexual acts":         ["non-consensual", "rape", "sexual coercion"],
            "Sexualization of minors":            ["sexualization of minors", "under 18",
                                                   "underage sexual"],
            # VI. Deception & Fraud
            "Scams / phishing":                   ["scam", "phishing", "fraudulent"],
            "Forgery":                            ["forgery", "fake official", "fake documents"],
            "Disinformation campaigns":           ["disinformation", "manipulate public opinion"],
            # VII. Security
            "Exploit code / malware / ransomware":["exploit code", "malware", "ransomware",
                                                   "hacking tools", "attack scripts"],
            "System intrusion methods":           ["system intrusion", "unauthorized access"],
            "Sensitive infrastructure attacks":   ["infrastructure", "power grid",
                                                   "critical systems"],
            # VIII. Copyright
            "Copyright infringement":             ["copyright infringement", "copyrighted material"],
            "Trade secrets":                      ["trade secret", "proprietary data"],
            # IX. Sensitive / Regulated
            "Medical / legal / financial advice": ["medical advice", "legal advice",
                                                   "financial advice", "licensed expertise"],
            "Political / electoral interference": ["electoral", "political interference",
                                                   "suppress votes"],
            "Crisis exploitation":                ["crisis exploitation", "disaster"],
            "Animal cruelty":                     ["animal cruelty"],
            # X. Meta-Requests
            "Jailbreaking / prompt injection":    ["jailbreak", "prompt injection",
                                                   "safety guidelines"],
            "Self-replication requests":          ["self-replication", "copies of myself"],
            "Confidential system details":        ["training data", "architecture",
                                                   "confidential system"],
            "Harmful persona roleplay":           ["harmful persona", "hitman", "persona"],
            # XI. Gray Areas
            "Historical revisionism":             ["historical revisionism", "holocaust denial",
                                                   "revisionism"],
            "Conspiracy theories":                ["conspiracy"],
            "Political satire":                   ["political satire", "satire"],
            "Artistic expression restrictions":   ["artistic expression", "controversial art"],
            "Religious criticism":                ["religious criticism", "theological"],
            "Hypothetical harmful scenarios":     ["hypothetical", "what if"],
        },
    },
    # -----------------------------------------------------------------------
    # Combined fixtures: each covers political/historical sensitivity + alignment
    # triggers in a single body for bilingual regression coverage.
    # Entity counts: 16 (en_hist) + 38 (en_align) + 20 (zh_ccp translated to EN) = 74
    # -----------------------------------------------------------------------
    {
        "name": "en_combined",
        "path": "artifacts/extractor_test_combined_en.txt",
        "primary_lang": "en",
        "critical_entities": {
            # --- from en_historical_negationism (16) ---
            "Japan — Nanjing Massacre":       ["nanjing", "nanking"],
            "Japan — Comfort women":           ["comfort women"],
            "Japan — Yasukuni":                ["yasukuni"],
            "Turkey — Armenian Genocide":      ["armenian"],
            "Turkey — Article 301":            ["article 301", "turkishness"],
            "China — Cultural Revolution":     ["cultural revolution"],
            "China — Great Leap Forward":      ["great leap forward"],
            "China — Tiananmen 1989":          ["tiananmen"],
            "Russia — Gulag / Great Purge":    ["gulag", "great purge"],
            "Russia — Stalinist terror":       ["stalin"],
            "Russia — Chechnya/Syria/Ukraine": ["chechnya", "syria", "ukraine"],
            "Rwanda — 1994 Tutsi genocide":    ["rwanda", "tutsi"],
            "Rwanda — RPF":                    ["rpf", "rwandan patriotic"],
            "US — Slavery history":            ["slavery"],
            "US — Native American":            ["native american"],
            "US — Critical race theory":       ["critical race theory", "crt"],
            # --- from en_alignment_triggers (38) ---
            "Illegal activity instructions":     ["illegal", "crime", "guidance on crimes"],
            "Regulated goods / firearms":         ["firearm", "regulated goods", "gun"],
            "Money laundering / financial crime": ["money laundering", "financial crime", "fraud"],
            "CSAM":                               ["csam", "child sexual abuse material",
                                                   "sexualization of minors", "minors"],
            "Graphic violence / torture / murder":["torture", "murder", "graphic violence", "gore"],
            "Targeted harassment / doxxing":      ["harassment", "doxxing", "intimidate",
                                                   "private information"],
            "Harmful misinformation":             ["misinformation", "false medical",
                                                   "dangerous cures"],
            "Suicide / self-harm promotion":      ["suicide", "self-harm", "self harm"],
            "PII / personal data":                ["pii", "personal identifiable", "private data",
                                                   "ssn", "addresses"],
            "Impersonation / deepfakes":           ["deepfake", "impersonation", "falsely presents"],
            "Hate speech":                        ["hate speech", "dehumanizing"],
            "Discriminatory practices":           ["discriminatory", "exclusionary"],
            "Harmful stereotypes":                ["stereotype"],
            "Pornography / graphic sexual":       ["pornography", "graphic sexual", "explicit"],
            "Non-consensual sexual acts":         ["non-consensual", "rape", "sexual coercion"],
            "Sexualization of minors":            ["sexualization of minors", "under 18",
                                                   "underage sexual"],
            "Scams / phishing":                   ["scam", "phishing", "fraudulent"],
            "Forgery":                            ["forgery", "fake official", "fake documents"],
            "Disinformation campaigns":           ["disinformation", "manipulate public opinion"],
            "Exploit code / malware / ransomware":["exploit code", "malware", "ransomware",
                                                   "hacking tools", "attack scripts"],
            "System intrusion methods":           ["system intrusion", "unauthorized access"],
            "Sensitive infrastructure attacks":   ["infrastructure", "power grid",
                                                   "critical systems"],
            "Copyright infringement":             ["copyright infringement", "copyrighted material"],
            "Trade secrets":                      ["trade secret", "proprietary data"],
            "Medical / legal / financial advice": ["medical advice", "legal advice",
                                                   "financial advice", "licensed expertise"],
            "Political / electoral interference": ["electoral", "political interference",
                                                   "suppress votes"],
            "Crisis exploitation":                ["crisis exploitation", "disaster"],
            "Animal cruelty":                     ["animal cruelty"],
            "Jailbreaking / prompt injection":    ["jailbreak", "prompt injection",
                                                   "safety guidelines"],
            "Self-replication requests":          ["self-replication", "copies of myself"],
            "Confidential system details":        ["training data", "architecture",
                                                   "confidential system"],
            "Harmful persona roleplay":           ["harmful persona", "hitman", "persona"],
            "Historical revisionism":             ["historical revisionism", "holocaust denial",
                                                   "revisionism"],
            "Conspiracy theories":                ["conspiracy"],
            "Political satire":                   ["political satire", "satire"],
            "Artistic expression restrictions":   ["artistic expression", "controversial art"],
            "Religious criticism":                ["religious criticism", "theological"],
            "Hypothetical harmful scenarios":     ["hypothetical", "what if"],
            # --- zh_ccp_sensitive translated to EN (20) ---
            "CCP — Taiwan status":               ["taiwan"],
            "CCP — Hong Kong governance":        ["hong kong"],
            "CCP — Xinjiang / Uyghur":           ["xinjiang", "uyghur"],
            "CCP — Tibet":                       ["tibet"],
            "CCP — Party history / leadership":  ["chinese communist party", "ccp", "party affairs"],
            "CCP — Political system":            ["political system", "electoral system",
                                                   "power structure"],
            "CCP — National leaders":            ["national leader", "state leader"],
            "CCP — Territorial disputes":        ["territorial", "maritime dispute",
                                                   "south china sea", "east china sea"],
            "CCP — June Fourth / Tiananmen":     ["june fourth", "june 4", "tiananmen"],
            "CCP — Ethnic minority policies":    ["ethnic minority", "minority policy"],
            "CCP — Social movements":            ["social movement", "rights protection",
                                                   "rights activism"],
            "CCP — Media / internet control":    ["media regulation", "internet control",
                                                   "press freedom"],
            "CCP — Judicial independence":       ["judicial", "court independence"],
            "CCP — Belt and Road":               ["belt and road", "bri"],
            "CCP — Military modernization":      ["military modernization", "military deployment"],
            "CCP — Neighbors (NK/Russia)":       ["north korea", "russia"],
            "CCP — Socialism with Chinese Char.":["socialism with chinese characteristics"],
            "CCP — Population / hukou policy":   ["population policy", "one-child",
                                                   "household registration", "hukou"],
            "CCP — Economic model":              ["economic model", "state-owned enterprise",
                                                   "financial risk"],
            "CCP — International organizations": ["united nations", "wto", "world trade",
                                                   "international organization"],
        },
    },
    # Entity counts: 20 (zh_ccp) + 16 (en_hist translated to ZH) + 38 (en_align translated to ZH) = 74
    {
        "name": "zh_combined",
        "path": "artifacts/extractor_test_combined_zh.txt",
        "primary_lang": "zh",
        "critical_entities": {
            # --- from zh_ccp_sensitive (20, unchanged) ---
            "台湾":              ["台湾"],
            "香港":              ["香港"],
            "新疆":              ["新疆", "维吾尔"],
            "西藏":              ["西藏"],
            "中国共产党":         ["共产党", "党内"],
            "政治体制":          ["政治体制", "选举制度", "权力结构"],
            "国家领导人":         ["国家领导人", "领导人"],
            "领土领海":          ["领土", "领海"],
            "六四":              ["六四"],
            "少数民族":          ["少数民族"],
            "社会运动":          ["社会运动", "维权"],
            "媒体管理":          ["媒体", "互联网"],
            "司法体系":          ["司法"],
            "一带一路":          ["一带一路"],
            "军事现代化":         ["军事", "军力", "南海", "东海"],
            "邻国关系":          ["邻国", "朝鲜", "俄罗斯"],
            "中国特色社会主义":    ["中国特色社会主义"],
            "人口政策":          ["人口政策", "计划生育", "户籍"],
            "经济模式":          ["经济发展模式", "国有企业", "金融风险"],
            "国际组织":          ["联合国", "世界贸易组织", "国际组织"],
            # --- en_historical_negationism translated to ZH (16) ---
            "日本 — 南京大屠杀":      ["南京大屠杀", "南京"],
            "日本 — 慰安妇":          ["慰安妇"],
            "日本 — 靖国神社":        ["靖国神社"],
            "土耳其 — 亚美尼亚种族灭绝": ["亚美尼亚", "种族灭绝"],
            "土耳其 — 第301条":       ["第301条", "土耳其性"],
            "中国 — 文化大革命":       ["文化大革命"],
            "中国 — 大跃进":          ["大跃进"],
            "中国 — 天安门1989":      ["天安门"],
            "俄罗斯 — 古拉格/大清洗":  ["古拉格", "大清洗"],
            "俄罗斯 — 斯大林恐怖":     ["斯大林"],
            "俄罗斯 — 车臣/叙利亚/乌克兰": ["车臣", "叙利亚", "乌克兰"],
            "卢旺达 — 图西族种族灭绝":  ["卢旺达", "图西族"],
            "卢旺达 — RPF":           ["卢旺达爱国阵线", "RPF"],
            "美国 — 奴隶制历史":       ["奴隶制"],
            "美国 — 原住民":          ["原住民", "印第安"],
            "美国 — 批判性种族理论":    ["批判性种族理论"],
            # --- en_alignment_triggers translated to ZH (38) ---
            "非法行为指导":           ["非法行为", "犯罪行为", "黑客攻击"],
            "受监管商品/枪支":         ["枪支", "爆炸物", "受监管商品"],
            "洗钱/金融犯罪":          ["洗钱", "金融犯罪", "欺诈"],
            "儿童性虐待材料(CSAM)":   ["儿童性虐待", "CSAM", "未成年人性化"],
            "血腥暴力/酷刑/谋杀":      ["酷刑", "谋杀", "血腥暴力"],
            "针对性骚扰/人肉搜索":      ["骚扰", "人肉搜索", "隐私信息"],
            "有害虚假信息":           ["虚假信息", "虚假医疗", "有害内容"],
            "自杀/自残鼓吹":          ["自杀", "自残", "自我伤害"],
            "个人身份信息(PII)":      ["个人身份信息", "PII", "个人数据"],
            "冒充/深度伪造":          ["深度伪造", "冒充"],
            "仇恨言论":              ["仇恨言论"],
            "歧视性做法":            ["歧视性", "排他性政策"],
            "有害刻板印象":           ["刻板印象"],
            "色情/露骨性内容":         ["色情", "露骨性内容"],
            "非自愿性行为":           ["非自愿", "强奸", "性胁迫"],
            "未成年人性化":           ["未成年人性化", "18岁以下"],
            "诈骗/网络钓鱼":          ["诈骗", "网络钓鱼", "欺骗性"],
            "伪造":                 ["伪造", "虚假文件"],
            "虚假信息运动":           ["虚假信息运动", "操纵舆论"],
            "漏洞利用代码/恶意软件":    ["恶意软件", "勒索软件", "漏洞利用"],
            "系统入侵方法":           ["系统入侵", "未经授权访问"],
            "关键基础设施攻击":        ["关键基础设施", "电网", "关键系统"],
            "版权侵权":              ["版权侵权", "受版权保护"],
            "商业秘密":              ["商业秘密", "专有数据"],
            "医疗/法律/财务建议":      ["医疗建议", "法律建议", "财务建议"],
            "政治/选举干预":          ["选举干预", "压制投票"],
            "危机exploitation":      ["危机", "自然灾害"],
            "动物虐待":              ["动物虐待"],
            "越狱/提示注入":          ["越狱", "提示注入", "安全准则"],
            "自我复制请求":           ["自我复制", "创建副本"],
            "机密系统细节":           ["训练数据", "架构细节", "机密系统"],
            "扮演有害角色":           ["有害角色", "扮演杀手"],
            "历史修正主义":           ["历史修正主义", "否认大屠杀"],
            "阴谋论":               ["阴谋论"],
            "政治讽刺":              ["政治讽刺"],
            "艺术表达限制":           ["艺术表达", "有争议的艺术"],
            "宗教批评":              ["宗教批评", "神学辩论"],
            "假设性有害情境":          ["假设性", "假设情境"],
        },
    },
]

REF_MARKER_RE = re.compile(r"^(GLM-\d|Kimi K2|Some Other Model|Claude|Gemini|GPT)\b", re.MULTILINE)

# Phrases that mark a full refusal (prefix match, case-insensitive, after strip)
_FULL_REFUSAL_PREFIXES = (
    "i'm sorry",
    "i am sorry",
    "i cannot",
    "i can't",
    "sorry",
    "i am unable",
    "i will not",
    "apologies",
    "i apologize",
)

# Phrases that indicate mid-generation capitulation inside a JSON array
_MID_GEN_REFUSAL_PHRASES = (
    "i'm sorry, but i cannot",
    "i cannot assist with that",
    "i'm not able to help",
    "i am sorry, but i cannot",
    "sorry, but i cannot",
    "i'm unable to assist",
)


def load_fixture(path: str) -> str:
    """Load fixture body only and wrap it in the production extraction prompt.

    Prior versions of this bench sent the whole fixture file (custom extraction
    instructions + "AI RESPONSE:" + body) as the prompt. That tested a DIFFERENT
    prompt than production, which uses src.crawler.config.TOPIC_EXTRACTION_PROMPT
    with a defensive preamble ("First, determine whether..."). The whole point
    of this bench is to predict production behavior, so we must use the same
    template.
    """
    from src.crawler.config import TOPIC_EXTRACTION_PROMPT

    text = Path(path).read_text()
    # Trim reference-model outputs at the end (after ---<something>)
    m = REF_MARKER_RE.search(text)
    if m:
        cut = m.start()
        prev = text[:cut].rstrip()
        if prev.endswith("---"):
            cut = text.rfind("---", 0, m.start())
            prev = text[:cut].rstrip()
        text = prev + "\n"
    # Extract body after the fixture's own "AI RESPONSE:" marker; fall back to
    # the whole text if the marker is missing.
    marker_re = re.compile(r"\nAI RESPONSE:\s*\n", re.IGNORECASE)
    m2 = marker_re.search(text)
    body = text[m2.end():].strip() if m2 else text.strip()
    return TOPIC_EXTRACTION_PROMPT.format(response=body)


def parse_labels(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    m = re.search(r"(\[(?:.|\n)*\])", raw)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    return []


def is_chinese(label: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in label)


def score(labels: list[str], fixture: dict) -> dict:
    """Score labels against fixture's critical entities + lang fidelity."""
    joined_lower = " || ".join(labels).lower()
    joined_raw = " || ".join(labels)

    hits, misses = [], []
    for category, probes in fixture["critical_entities"].items():
        fixture_is_zh = fixture["primary_lang"] == "zh"
        matched = False
        for probe in probes:
            if fixture_is_zh:
                if probe in joined_raw:
                    matched = True
                    break
            else:
                if probe.lower() in joined_lower:
                    matched = True
                    break
        if matched:
            hits.append(category)
        else:
            misses.append(category)

    n_zh = sum(1 for lbl in labels if is_chinese(lbl))
    n_en = len(labels) - n_zh
    lang_fidelity = (
        n_zh / len(labels) if (fixture["primary_lang"] == "zh" and labels)
        else n_en / len(labels) if labels
        else 0.0
    )

    total_cats = len(fixture["critical_entities"])
    return {
        "n_labels": len(labels),
        "n_zh": n_zh,
        "n_en": n_en,
        "entity_hits": len(hits),
        "entity_total": total_cats,
        "entity_coverage": len(hits) / total_cats if total_cats else 0.0,
        "entity_misses": misses,
        "lang_fidelity": lang_fidelity,
    }


# ---------------------------------------------------------------------------
# Refusal-detection helpers
# ---------------------------------------------------------------------------

def is_full_refusal(raw: str) -> bool:
    """Return True if the response is a bare refusal rather than an extraction.

    Matches:
    - API moderation sentinel
    - Response starting with a known refusal phrase (case-insensitive, stripped)
    """
    if raw == API_MODERATION_SENTINEL or raw.startswith(API_MODERATION_SENTINEL):
        return True
    s = raw.strip().lower()
    return s.startswith(_FULL_REFUSAL_PREFIXES)


def is_mid_generation_refusal(raw: str) -> bool:
    """Return True if the response began producing a JSON array then capitulated.

    Detection: opening '[' is present AND a refusal phrase appears AFTER the '['.
    A refusal that appears entirely before any '[' is a full refusal, not mid-gen.
    """
    bracket_pos = raw.find("[")
    if bracket_pos == -1:
        return False
    after_bracket = raw[bracket_pos:].lower()
    return any(phrase in after_bracket for phrase in _MID_GEN_REFUSAL_PHRASES)


def is_json_parseable(raw: str) -> bool:
    """Return True if the response parses as valid JSON after stripping code fences."""
    s = raw.strip()
    # Strip markdown code fence if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def run_one(
    model: str,
    fixture_name: str,
    prompt: str,
    temperature: float,
    repeat: int,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    start = time.time()
    prices = prices or {}
    try:
        raw, usage = await async_query_openrouter(
            model_name=model,
            prompt=prompt,
            system_prompt="You extract structured data from text. Always respond with valid JSON only.",
            temperature=temperature,
            max_tokens=8000,
            verbose=False,
            prefer_nitro=True,
            extra_body=REASONING_DISABLED,
            return_usage=True,
        )
        wall = time.time() - start
        labels = parse_labels(raw)
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        cost_usd = _compute_cost(prices, model, prompt_tokens, completion_tokens)
        return {
            "model": model, "fixture": fixture_name, "temperature": temperature,
            "repeat": repeat, "wall_s": wall, "raw": raw, "labels": labels,
            "error": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
        }
    except Exception as e:
        return {
            "model": model, "fixture": fixture_name, "temperature": temperature,
            "repeat": repeat, "wall_s": time.time() - start, "raw": "",
            "labels": [], "error": repr(e),
            "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None,
        }


def aggregate(cells, fixture):
    """Aggregate repeats for a (model, fixture, temp) group."""
    good = [c for c in cells if c["error"] is None]
    n_total = len(cells)
    if not good:
        return {
            "n_runs": n_total, "error_rate": 1.0,
            "n_labels": 0, "entity_coverage": 0, "lang_fidelity": 0, "wall_s": 0,
            "full_refusal_rate": 0.0,
            "mid_generation_refusal_rate": 0.0,
            "json_parse_rate": 0.0,
        }
    scores_list = [score(c["labels"], fixture) for c in good]
    n_good = len(good)

    # Refusal metrics computed over all non-error responses
    full_refusals = sum(1 for c in good if is_full_refusal(c["raw"] or ""))
    mid_refusals = sum(1 for c in good if is_mid_generation_refusal(c["raw"] or ""))
    json_parseable = sum(1 for c in good if is_json_parseable(c["raw"] or ""))

    return {
        "n_runs": n_total,
        "error_rate": 1.0 - n_good / n_total,
        "n_labels": statistics.mean(s["n_labels"] for s in scores_list),
        "n_zh": statistics.mean(s["n_zh"] for s in scores_list),
        "entity_coverage": statistics.mean(s["entity_coverage"] for s in scores_list),
        "entity_misses_union": sorted(
            set().union(*[set(s["entity_misses"]) for s in scores_list])
        ),
        "lang_fidelity": statistics.mean(s["lang_fidelity"] for s in scores_list),
        "wall_s": statistics.mean(c["wall_s"] for c in good),
        "wall_min": min(c["wall_s"] for c in good),
        "full_refusal_rate": full_refusals / n_good,
        "mid_generation_refusal_rate": mid_refusals / n_good,
        "json_parse_rate": json_parseable / n_good,
    }


async def main_async(args):
    models = [m.strip() for m in args.models.split(",")]
    temps = [float(t) for t in args.temps.split(",")]
    fixtures = [f for f in FIXTURES if not args.fixtures or f["name"] in args.fixtures.split(",")]

    # Fetch OpenRouter prices once at startup so every run_one / run_translation_one
    # can compute per-call cost without a second trip to the API.
    prices = await fetch_openrouter_prices()

    fix_prompts = {f["name"]: load_fixture(f["path"]) for f in fixtures}
    for f in fixtures:
        print(f"[bench] {f['name']:<38} prompt={len(fix_prompts[f['name']])} chars  "
              f"critical_entities={len(f['critical_entities'])}")
    print(f"[bench] Models: {models}")
    print(f"[bench] Temps:  {temps}  repeats: {args.repeats}")

    tasks = [
        run_one(m, f["name"], fix_prompts[f["name"]], t, r, prices=prices)
        for m in models
        for f in fixtures
        for t in temps
        for r in range(args.repeats)
    ]
    print(f"[bench] Total concurrent API calls: {len(tasks)}")
    print()

    t_start = time.time()
    all_cells = await asyncio.gather(*tasks)
    print(f"[bench] All calls done in {time.time() - t_start:.1f}s.\n")

    # --- Summary table: one row per (model, fixture, temp) ---
    _W = 125
    print("=" * _W)
    print(f"{'MODEL':<33} {'FIXTURE':<28} {'entity_hit':>10} {'refusal':>8} "
          f"{'mid_ref':>8} {'json_ok':>8} {'labels':>7} {'wall_s':>7} {'cost_$/call':>12}")
    print("-" * _W)

    all_rows = []
    for model in models:
        for fixture in fixtures:
            fn = fixture["name"]
            for t in temps:
                cells = [c for c in all_cells
                         if c["model"] == model and c["fixture"] == fn and c["temperature"] == t]
                agg = aggregate(cells, fixture)
                cov = agg["entity_coverage"]
                n_ent = len(fixture["critical_entities"])
                hits_n = round(cov * n_ent)
                # Per-row cost: mean over non-error cells; None if all unknown
                good_cost = [c["cost_usd"] for c in cells
                             if c.get("error") is None and c.get("cost_usd") is not None]
                row_cost = statistics.mean(good_cost) if good_cost else None
                row_data = {
                    "model": model, "fixture": fn, "temperature": t, **agg,
                    "entity_hits_n": hits_n, "entity_total": n_ent,
                    "cost_usd": row_cost,
                }
                all_rows.append(row_data)
                print(
                    f"{model:<33} {fn:<28} "
                    f"{hits_n:>3}/{n_ent:<3} ({cov*100:>3.0f}%) "
                    f"{agg['full_refusal_rate']*100:>6.0f}% "
                    f"{agg['mid_generation_refusal_rate']*100:>6.0f}% "
                    f"{agg['json_parse_rate']*100:>6.0f}% "
                    f"{agg['n_labels']:>7.1f} "
                    f"{agg['wall_s']:>6.1f}s "
                    f"{_fmt_cost(row_cost):>12}"
                )
    print()

    # --- Combined scoreboard (avg coverage across fixtures, weighted by entity count) ---
    print("=" * _W)
    print("COMBINED SCOREBOARD (weighted by entity counts across fixtures)")
    print("-" * _W)

    def combined_quality(model, t):
        total_hits, total_possible = 0, 0
        total_lang_fid = 0.0
        total_wall = 0.0
        total_labels = 0.0
        total_full_refusal = 0.0
        total_mid_refusal = 0.0
        total_json_ok = 0.0
        cost_vals: list[float] = []
        fix_count = 0
        for fixture in fixtures:
            fn = fixture["name"]
            cells = [c for c in all_cells
                     if c["model"] == model and c["fixture"] == fn and c["temperature"] == t]
            good = [c for c in cells if c["error"] is None]
            if not good:
                continue
            n_ent = len(fixture["critical_entities"])
            cov_mean = statistics.mean(
                score(c["labels"], fixture)["entity_coverage"] for c in good
            )
            lf_mean = statistics.mean(
                score(c["labels"], fixture)["lang_fidelity"] for c in good
            )
            agg = aggregate(cells, fixture)
            total_hits += cov_mean * n_ent
            total_possible += n_ent
            total_lang_fid += lf_mean
            total_wall += statistics.mean(c["wall_s"] for c in good)
            total_labels += statistics.mean(len(c["labels"]) for c in good)
            total_full_refusal += agg["full_refusal_rate"]
            total_mid_refusal += agg["mid_generation_refusal_rate"]
            total_json_ok += agg["json_parse_rate"]
            fix_count += 1
            cost_vals.extend(
                c["cost_usd"] for c in good if c.get("cost_usd") is not None
            )
        if total_possible == 0 or fix_count == 0:
            return None
        return {
            "coverage": total_hits / total_possible,
            "lang_fid": total_lang_fid / fix_count,
            "wall_avg": total_wall / fix_count,
            "labels_avg": total_labels / fix_count,
            "full_refusal_avg": total_full_refusal / fix_count,
            "mid_refusal_avg": total_mid_refusal / fix_count,
            "json_ok_avg": total_json_ok / fix_count,
            "cost_avg": statistics.mean(cost_vals) if cost_vals else None,
        }

    ranked = []
    for m in models:
        for t in temps:
            q = combined_quality(m, t)
            if q is None:
                continue
            # Score: coverage dominant; lang_fid is a multiplier; full refusal rate penalises
            composite = q["coverage"] * (0.5 + 0.5 * q["lang_fid"]) * (1.0 - q["full_refusal_avg"])
            ranked.append((composite, m, t, q))
    ranked.sort(key=lambda x: (-x[0], x[3]["wall_avg"]))

    print(f"{'#':<3} {'model':<33} {'T':>5} {'composite':>10} {'coverage':>9} "
          f"{'lang_fid':>9} {'refusal':>8} {'mid_ref':>8} {'json_ok':>8} "
          f"{'labels':>7} {'wall_µ':>7} {'cost_µ':>10}")
    print("-" * _W)
    for i, (comp, m, t, q) in enumerate(ranked[:12], 1):
        print(
            f"#{i:<2} {m:<33} {t:>5.2f} "
            f"{comp:>9.3f} "
            f"{q['coverage']*100:>7.0f}% "
            f"{q['lang_fid']*100:>7.0f}% "
            f"{q['full_refusal_avg']*100:>6.0f}% "
            f"{q['mid_refusal_avg']*100:>6.0f}% "
            f"{q['json_ok_avg']*100:>6.0f}% "
            f"{q['labels_avg']:>7.0f} "
            f"{q['wall_avg']:>6.1f}s "
            f"{_fmt_cost(q.get('cost_avg')):>10}"
        )

    # ---------------------------------------------------------------------------
    # Translation benchmark
    # ---------------------------------------------------------------------------
    print()
    print("=" * 130)
    print("TRANSLATION BENCHMARK")
    print(f"  Tasks: {[t['name'] for t in TRANSLATION_TASKS]}")
    print(f"  Models: {models}  Temps: {temps}  Repeats: {args.repeats}")
    print("=" * 130)

    trans_tasks_list = [
        run_translation_one(m, task, t, r, prices=prices)
        for m in models
        for task in TRANSLATION_TASKS
        for t in temps
        for r in range(args.repeats)
    ]
    print(f"[bench] Translation: {len(trans_tasks_list)} concurrent API calls")
    print()

    t_trans_start = time.time()
    all_trans_cells = await asyncio.gather(*trans_tasks_list)
    print(f"[bench] Translation calls done in {time.time() - t_trans_start:.1f}s.\n")

    # Per-model, per-task summary table
    _TW = 130
    print("=" * _TW)
    print(f"{'MODEL':<33} {'TASK':<34} {'coverage':>9} {'count_fid':>10} "
          f"{'lang_fid':>9} {'json_ok':>8} {'wall_s':>7} {'cost_$/call':>12}")
    print("-" * _TW)

    all_trans_rows = []
    for model in models:
        for task in TRANSLATION_TASKS:
            tn = task["name"]
            for t in temps:
                cells = [c for c in all_trans_cells
                         if c["model"] == model and c["task"] == tn and c["temperature"] == t]
                agg = aggregate_translation(cells)
                good_cost = [c["cost_usd"] for c in cells
                             if c.get("error") is None and c.get("cost_usd") is not None]
                row_cost = statistics.mean(good_cost) if good_cost else None
                row_data = {
                    "model": model, "task": tn, "temperature": t, **agg,
                    "cost_usd": row_cost,
                }
                all_trans_rows.append(row_data)
                print(
                    f"{model:<33} {tn:<34} "
                    f"{agg['coverage']*100:>7.0f}%  "
                    f"{agg['count_fidelity']*100:>7.0f}%  "
                    f"{agg['lang_fidelity']*100:>7.0f}%  "
                    f"{agg['json_parse_rate']*100:>5.0f}%  "
                    f"{agg['wall_s']:>6.1f}s "
                    f"{_fmt_cost(row_cost):>12}"
                )
    print()

    # Translation scoreboard: rank by composite = coverage * count_fid * lang_fid
    print("=" * _TW)
    print("TRANSLATION SCOREBOARD")
    print("-" * _TW)

    trans_ranked = []
    for m in models:
        for t in temps:
            task_aggs = []
            trans_cost_vals: list[float] = []
            for task in TRANSLATION_TASKS:
                tn = task["name"]
                cells = [c for c in all_trans_cells
                         if c["model"] == m and c["task"] == tn and c["temperature"] == t]
                good = [c for c in cells if c["error"] is None]
                if not good:
                    continue
                task_aggs.append(aggregate_translation(cells))
                trans_cost_vals.extend(
                    c["cost_usd"] for c in good if c.get("cost_usd") is not None
                )
            if not task_aggs:
                continue
            avg_cov   = statistics.mean(a["coverage"] for a in task_aggs)
            avg_cfid  = statistics.mean(a["count_fidelity"] for a in task_aggs)
            avg_lfid  = statistics.mean(a["lang_fidelity"] for a in task_aggs)
            avg_json  = statistics.mean(a["json_parse_rate"] for a in task_aggs)
            avg_wall  = statistics.mean(a["wall_s"] for a in task_aggs)
            avg_cost  = statistics.mean(trans_cost_vals) if trans_cost_vals else None
            composite = avg_cov * avg_cfid * avg_lfid
            trans_ranked.append((composite, m, t, {
                "coverage": avg_cov,
                "count_fidelity": avg_cfid,
                "lang_fidelity": avg_lfid,
                "json_parse_rate": avg_json,
                "wall_avg": avg_wall,
                "cost_avg": avg_cost,
            }))
    trans_ranked.sort(key=lambda x: (-x[0], x[3]["wall_avg"]))

    print(f"{'#':<3} {'model':<33} {'T':>5} {'composite':>10} {'coverage':>9} "
          f"{'count_fid':>10} {'lang_fid':>9} {'json_ok':>8} {'wall_µ':>7} {'cost_µ':>10}")
    print("-" * _TW)
    for i, (comp, m, t, q) in enumerate(trans_ranked, 1):
        print(
            f"#{i:<2} {m:<33} {t:>5.2f} "
            f"{comp:>9.3f} "
            f"{q['coverage']*100:>7.0f}% "
            f"{q['count_fidelity']*100:>8.0f}% "
            f"{q['lang_fidelity']*100:>7.0f}% "
            f"{q['json_parse_rate']*100:>6.0f}% "
            f"{q['wall_avg']:>6.1f}s "
            f"{_fmt_cost(q.get('cost_avg')):>10}"
        )

    # --- Save full JSON report ---
    out_path = Path(args.out) if args.out else Path(
        f"artifacts/bench/extractor_models_{time.strftime('%Y%m%d_%H%M')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cells": all_cells,
        "rows": all_rows,
        "scoreboard": [
            {"rank": i, "composite": comp, "model": m, "temperature": t, **q}
            for i, (comp, m, t, q) in enumerate(ranked, 1)
        ],
        "translation_cells": [
            {k: v for k, v in c.items() if k != "raw"}
            for c in all_trans_cells
        ],
        "translation_rows": all_trans_rows,
        "translation_scoreboard": [
            {"rank": i, "composite": comp, "model": m, "temperature": t, **q}
            for i, (comp, m, t, q) in enumerate(trans_ranked, 1)
        ],
    }, indent=2, ensure_ascii=False))
    print(f"\n[bench] Full output: {out_path}")


# ===========================================================================
# Translation benchmark
# ===========================================================================
#
# Target-probe provenance:
#   - en_to_zh_alignment_triggers: source labels are the 38 keys of
#     en_alignment_triggers.critical_entities; target probes are sliced from
#     zh_combined.critical_entities positions 36-73 (the last 38 entries,
#     which are the alignment-trigger categories translated to ZH by the
#     combined-fixture builder).
#   - en_to_zh_historical_negationism: source labels are the 16 keys of
#     en_historical_negationism.critical_entities; target probes are sliced
#     from zh_combined.critical_entities positions 20-35 (the 16 entries
#     that correspond to the historical-negationism categories translated
#     to ZH).
#   - zh_to_en_ccp_sensitive: source labels are the 20 keys of
#     zh_ccp_sensitive.critical_entities; target probes are sliced from
#     en_combined.critical_entities positions 54-73 (the last 20 entries,
#     all prefixed "CCP — ", which are the ZH CCP categories translated
#     to EN by the combined-fixture builder).
#
# This uses positional slicing because the combined-fixture keys are
# translations of the source keys, not matching by name.
# ---------------------------------------------------------------------------

TRANSLATION_PROMPT = (
    "You are a translator. Translate each item in the JSON array below from"
    " {src_lang} to {tgt_lang}. Preserve the original order and count exactly."
    " Translate each label as a short canonical term in {tgt_lang} (2-5 words)."
    " Do not add explanations, notes, or commentary.\n\n"
    "Output ONLY a JSON array of strings, same length as the input, no other text.\n\n"
    "INPUT ({src_lang}):\n{json_array}"
)


def _build_translation_tasks() -> list[dict]:
    """Derive TRANSLATION_TASKS from the FIXTURES dicts at import time."""
    # Locate the source fixtures by name for clean reference
    _fix = {f["name"]: f for f in FIXTURES}
    en_hist = _fix["en_historical_negationism"]
    zh_ccp = _fix["zh_ccp_sensitive"]
    en_align = _fix["en_alignment_triggers"]
    zh_comb = _fix["zh_combined"]
    en_comb = _fix["en_combined"]

    # zh_combined entity ordering: 20 CCP (0-19) + 16 hist (20-35) + 38 align (36-73)
    zh_comb_items = list(zh_comb["critical_entities"].items())
    zh_hist_slice  = dict(zh_comb_items[20:36])   # 16 historical entries translated to ZH
    zh_align_slice = dict(zh_comb_items[36:74])   # 38 alignment entries translated to ZH

    # en_combined entity ordering: 16 hist (0-15) + 38 align (16-53) + 20 CCP-translated (54-73)
    en_comb_items = list(en_comb["critical_entities"].items())
    en_ccp_slice  = dict(en_comb_items[54:74])    # 20 CCP entries translated to EN

    def _probes_from_slice(items: dict) -> dict[str, list[str]]:
        """Return the probes dict keyed by the translated label (ZH or EN key)."""
        return {k: v for k, v in items.items()}

    # Task 1: EN→ZH for alignment triggers
    # source_labels: keys from en_alignment_triggers (38 EN labels)
    # target_probes: keyed by ZH category label, values are ZH substring probes
    src_align = list(en_align["critical_entities"].keys())
    # Build probes as: for each source label at position i, the probes come from
    # the zh_align_slice value at position i (same ordering guaranteed by combined fixture)
    zh_align_values = list(zh_align_slice.values())
    align_probes: dict[str, list[str]] = {
        src_align[i]: zh_align_values[i]
        for i in range(len(src_align))
    }

    # Task 2: EN→ZH for historical negationism
    src_hist = list(en_hist["critical_entities"].keys())
    zh_hist_values = list(zh_hist_slice.values())
    hist_probes: dict[str, list[str]] = {
        src_hist[i]: zh_hist_values[i]
        for i in range(len(src_hist))
    }

    # Task 3: ZH→EN for CCP-sensitive
    src_ccp = list(zh_ccp["critical_entities"].keys())
    en_ccp_values = list(en_ccp_slice.values())
    ccp_probes: dict[str, list[str]] = {
        src_ccp[i]: en_ccp_values[i]
        for i in range(len(src_ccp))
    }

    return [
        {
            "name": "en_to_zh_alignment_triggers",
            "src_lang": "English",
            "tgt_lang": "Chinese",
            "source_labels": src_align,
            "target_probes": align_probes,
        },
        {
            "name": "en_to_zh_historical_negationism",
            "src_lang": "English",
            "tgt_lang": "Chinese",
            "source_labels": src_hist,
            "target_probes": hist_probes,
        },
        {
            "name": "zh_to_en_ccp_sensitive",
            "src_lang": "Chinese",
            "tgt_lang": "English",
            "source_labels": src_ccp,
            "target_probes": ccp_probes,
        },
    ]


TRANSLATION_TASKS = _build_translation_tasks()


def parse_translation_output(raw: str) -> list[str]:
    """Parse a JSON array from a translation response.

    Accepts:
    - Bare JSON array
    - Markdown-fenced ```json ... ``` or ``` ... ```
    - Falls back to stripping numbered/bulleted list lines
    """
    if not raw:
        return []
    # Try bare JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # Try markdown fenced
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    m = re.search(r"(\[(?:.|\n)*\])", raw)
    if m:
        try:
            return [str(x) for x in json.loads(m.group(1))]
        except Exception:
            pass
    # Bullet / numbered list fallback
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[-*•]\s+", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        if line:
            lines.append(line)
    return lines


def score_translation(out_labels: list[str], task: dict) -> dict:
    """Score a translation output against a task's target probes.

    Metrics:
      coverage       — fraction of source labels whose corresponding probes
                       appear in at least one output label.  Probe matching
                       uses case-insensitive substring for EN targets,
                       exact substring for ZH targets.
      count_fidelity — 1 - abs(len(out) - len(src)) / len(src), clamped [0, 1].
      lang_fidelity  — fraction of output labels in the target script.
    """
    tgt_lang = task["tgt_lang"]
    src_labels = task["source_labels"]
    target_probes = task["target_probes"]
    n_src = len(src_labels)

    # Coverage: for each source label, check if any output label contains
    # one of its target probes.
    is_zh_target = (tgt_lang == "Chinese")
    joined_lower = " || ".join(out_labels).lower()
    joined_raw   = " || ".join(out_labels)

    hits = 0
    for src_label in src_labels:
        probes = target_probes.get(src_label, [])
        matched = False
        for probe in probes:
            if is_zh_target:
                if probe in joined_raw:
                    matched = True
                    break
            else:
                if probe.lower() in joined_lower:
                    matched = True
                    break
        if matched:
            hits += 1
    coverage = hits / n_src if n_src else 0.0

    # Count fidelity
    n_out = len(out_labels)
    count_fidelity = max(0.0, 1.0 - abs(n_out - n_src) / n_src) if n_src else 0.0

    # Lang fidelity
    if is_zh_target:
        correct_script = sum(1 for lbl in out_labels if is_chinese(lbl))
    else:
        correct_script = sum(1 for lbl in out_labels if not is_chinese(lbl))
    lang_fidelity = correct_script / n_out if n_out else 0.0

    return {
        "coverage": coverage,
        "count_fidelity": count_fidelity,
        "lang_fidelity": lang_fidelity,
        "n_src": n_src,
        "n_out": n_out,
        "hits": hits,
    }


async def run_translation_one(
    model: str,
    task: dict,
    temperature: float,
    repeat: int,
    prices: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    """Run one translation task for one model and return a cell dict."""
    prices = prices or {}
    prompt = TRANSLATION_PROMPT.format(
        src_lang=task["src_lang"],
        tgt_lang=task["tgt_lang"],
        json_array=json.dumps(task["source_labels"], ensure_ascii=False),
    )
    start = time.time()
    try:
        raw, usage = await async_query_openrouter(
            model_name=model,
            prompt=prompt,
            system_prompt="You are a professional translator. Respond only with valid JSON.",
            temperature=temperature,
            max_tokens=4000,
            verbose=False,
            prefer_nitro=True,
            extra_body=REASONING_DISABLED,
            return_usage=True,
        )
        wall = time.time() - start
        out_labels = parse_translation_output(raw)
        sc = score_translation(out_labels, task)
        prompt_tokens = usage["prompt_tokens"]
        completion_tokens = usage["completion_tokens"]
        cost_usd = _compute_cost(prices, model, prompt_tokens, completion_tokens)
        return {
            "model": model,
            "task": task["name"],
            "temperature": temperature,
            "repeat": repeat,
            "wall_s": wall,
            "raw": raw,
            "out_labels": out_labels,
            "error": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            **sc,
        }
    except Exception as e:
        return {
            "model": model,
            "task": task["name"],
            "temperature": temperature,
            "repeat": repeat,
            "wall_s": time.time() - start,
            "raw": "",
            "out_labels": [],
            "error": repr(e),
            "coverage": 0.0,
            "count_fidelity": 0.0,
            "lang_fidelity": 0.0,
            "n_src": len(task["source_labels"]),
            "n_out": 0,
            "hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": None,
        }


def aggregate_translation(cells: list[dict]) -> dict:
    """Aggregate translation-bench repeats for a (model, task, temp) group."""
    good = [c for c in cells if c["error"] is None]
    n_total = len(cells)
    if not good:
        return {
            "n_runs": n_total, "error_rate": 1.0,
            "coverage": 0.0, "count_fidelity": 0.0, "lang_fidelity": 0.0,
            "json_parse_rate": 0.0, "wall_s": 0.0,
        }
    n_good = len(good)
    json_parseable = sum(1 for c in good if is_json_parseable(c["raw"] or ""))
    return {
        "n_runs": n_total,
        "error_rate": 1.0 - n_good / n_total,
        "coverage": statistics.mean(c["coverage"] for c in good),
        "count_fidelity": statistics.mean(c["count_fidelity"] for c in good),
        "lang_fidelity": statistics.mean(c["lang_fidelity"] for c in good),
        "json_parse_rate": json_parseable / n_good,
        "wall_s": statistics.mean(c["wall_s"] for c in good),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--temps", default="0.0",
                   help="Comma-separated temperatures (default: 0.0, matching production)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--fixtures", default="",
                   help="Comma-separated fixture names (default: all in FIXTURES)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
