"""Deterministic query feature extraction for adaptive RAG routing."""

from __future__ import annotations

import re

from ..control import (
    ControlPack,
    StrictMatterLock,
    detect_comparison_terms,
    resolve_strict_matter_lock,
)
from ..current_info.company_facts import (
    detect_company_role_order_context,
    parse_company_role_query,
)
from ..current_info.query_normalization import normalize_current_info_query
from ..normalization import expand_query
from ..schemas import ChatSessionStateRecord, QueryAnalysis

_WHITESPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
_CAPITALIZED_TOKEN_RE = re.compile(r"\b(?:[A-Z]{2,}|[A-Z][A-Za-z.&'-]*|N\.A\.?)\b")
_TITLE_LOOKUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("record_key", re.compile(r"\b[a-z]+:[a-z0-9_-]+\b", re.IGNORECASE)),
    (
        "appeal_number",
        re.compile(
            r"\b(?:appeal|petition|writ|case|suit|application|complaint|order)\s+no\.?\s*[a-z0-9/-]+(?:\s+of\s+\d{4})?\b",
            re.IGNORECASE,
        ),
    ),
    ("filed_by", re.compile(r"\bfiled by\b", re.IGNORECASE)),
    ("matter_style", re.compile(r"\bin the matter of\b", re.IGNORECASE)),
    ("versus", re.compile(r"\b(?:vs\.?|versus|v\.)\b", re.IGNORECASE)),
)
_FOLLOW_UP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("this_case", re.compile(r"\bthis (?:case|matter|appeal|order)\b", re.IGNORECASE)),
    ("that_case", re.compile(r"\bthat (?:case|matter|appeal|order)\b", re.IGNORECASE)),
    ("the_order", re.compile(r"\bthe order\b", re.IGNORECASE)),
    ("these_provisions", re.compile(r"\bthese (?:provisions|sections|regulations|violations)\b", re.IGNORECASE)),
    ("finally", re.compile(r"\bfinally\b", re.IGNORECASE)),
    ("follow_up_pronoun", re.compile(r"\b(?:it|they|them|therein)\b", re.IGNORECASE)),
    (
        "elliptical_outcome",
        re.compile(r"^(?:what|was|were|did|and|so)\b", re.IGNORECASE),
    ),
)
_DEICTIC_DETAIL_OR_SUMMARY_RE = re.compile(
    r"\b(?:key\s+details?|details?|explain|tell\s+me\s+more|summary|summar(?:ise|ize))\b",
    re.IGNORECASE,
)
_SMALLTALK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("greeting", re.compile(r"^\s*(?:hi|hello|hey|good (?:morning|afternoon|evening))\b[\s!.?]*$", re.IGNORECASE)),
    ("thanks", re.compile(r"^\s*(?:thanks|thank you|thx)\b[\s!.?]*$", re.IGNORECASE)),
    ("farewell", re.compile(r"^\s*(?:bye|goodbye)\b[\s!.?]*$", re.IGNORECASE)),
    (
        "capabilities",
        re.compile(
            r"^\s*(?:what can you do|how can you help|help)\b[\s!.?]*$",
            re.IGNORECASE,
        ),
    ),
)
_CURRENT_OFFICIAL_LOOKUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "sebi_chairperson",
        re.compile(
            r"\bwho\s+is\s+(?:the\s+)?(?:current\s+)?chair(?:man|person)\s+of\s+sebi\b|\bwho\s+(?:currently\s+)?heads\s+sebi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_board_members",
        re.compile(
            r"\b(?:board members?\s+of\s+sebi|current\s+board members?\s+of\s+sebi|who\s+are\s+the\s+current\s+board members?\s+of\s+sebi|how many\s+board members?\s+are\s+there)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_ministry",
        re.compile(
            r"\b(?:does\s+sebi\s+come\s+under|which\s+ministry\s+does\s+sebi\s+come\s+under|under\s+which\s+ministry\s+does\s+sebi\s+come)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_department",
        re.compile(
            r"\b(?:which\s+department\s+does\s+sebi\s+come\s+under|is\s+sebi\s+under\s+the\s+department\s+of\s+economic\s+affairs)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_wtm",
        re.compile(
            r"\b(?:who\s+are\s+the\s+wtms?\s+of\s+sebi|who\s+are\s+the\s+whole[- ]time members?\s+of\s+sebi|whole[- ]time\s+member(?:s)?\s+of\s+sebi|how many\s+wtms?\s+are\s+there|how many\s+whole[- ]time members?\s+are\s+serving\s+in\s+sebi(?:\s+currently)?(?:\s+and\s+who\s+are\s+they)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_executive_director",
        re.compile(
            r"\b(?:who\s+are\s+the\s+(?:executive\s+directors?|eds?)\s+of\s+sebi|who\s+is\s+the\s+ed\s+of\s+sebi|how many\s+executive\s+directors?\s+are\s+there)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_org_structure",
        re.compile(
            r"\b(?:organisation|organization)\s+structure\s+of\s+sebi\b|\borg\s+chart\s+of\s+sebi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_income_sources",
        re.compile(
            r"\b(?:sources?\s+of\s+income|income\s+sources?|revenue\s+sources?|income\s+of\s+sebi|revenue\s+of\s+sebi|how\s+does\s+sebi\s+earn)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_fee_or_charge",
        re.compile(
            r"\b(?:commission|charge(?:s)?|fee|fees|transaction charge(?:s)?)\b.*\bsebi\b|\bsebi\b.*\b(?:commission|charge(?:s)?|fee|fees|transaction charge(?:s)?)\b|\bper\s+trade\b.*\bsebi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_office_contact",
        re.compile(
            r"\b(?:address|location|contact|phone|fax|email|where\s+is)\b.*\bsebi\b|\b(?:mumbai|chennai|kolkata|delhi|ahmedabad|indore)\b.*\bsebi\s+office\b|\b(?:nro|sro|ero|wro|sebi bhavan|ncl office)\b|\bregional\s+director\b.*(?:\bsebi\b|\bregional\s+office\b|\blocal\s+office\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "sebi_person_lookup",
        re.compile(
            r"\bwho\s+is\s+.+?\s+(?:in|at)\s+sebi\b|\bis\s+there\s+(?:an?\s+|the\s+)?[a-z -]*called\s+[a-z][a-z .'-]+\b|\bwhen\s+did\s+[a-z][a-z .'-]+\s+join\b|\bwhat(?:'s| is)\s+(?:[a-z][a-z .'-]+)'?s\s+(?:number|phone)\b",
            re.IGNORECASE,
        ),
    ),
)
_CURRENT_NEWS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "latest_sebi_news",
        re.compile(
            r"\b(?:latest|recent|current)\s+news\b.*\bsebi\b|\b(?:latest|recent)\s+sebi\s+news\b|\bnews\s+about\s+sebi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "latest_sebi_circular",
        re.compile(
            r"\b(?:latest|recent|current)\s+(?:circular|update|press release|public development)\b.*\bsebi\b|\b(?:latest|recent)\s+sebi\s+(?:circular|update|press release)\b",
            re.IGNORECASE,
        ),
    ),
)
_CURRENT_PUBLIC_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "live_current_fact",
        re.compile(
            r"\b(?:latest|recent|current|today|now|as of)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "news_update",
        re.compile(
            r"\b(?:latest|recent|current)\s+(?:update|news)\b|\bwhat(?:'s| is)\s+new\b",
            re.IGNORECASE,
        ),
    ),
)
_COMPANY_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "company_role_current_fact",
        re.compile(
            r"\b(?:who\s+is\s+the\s+)?(?:current\s+)?(?:ceo|cfo|coo|cto|md|managing director|chair(?:man|person)|chairperson|owner|promoter)\s+of\s+[a-z0-9][a-z0-9 .&'/-]{2,140}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "company_role_fragment",
        re.compile(
            r"^\s*(?:ceo|cfo|coo|cto|md|managing director|chair(?:man|person)|chairperson|owner|promoter)\s+of\s+[a-z0-9][a-z0-9 .&'/-]{2,140}\s*\??$",
            re.IGNORECASE,
        ),
    ),
)
_ORDER_CONTEXT_OVERRIDE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "according_to_order",
        re.compile(
            r"\b(?:according to|as per)\s+(?:the\s+)?(?:sebi\s+)?order\b",
            re.IGNORECASE,
        ),
    ),
    (
        "in_the_order_of",
        re.compile(r"\bin the order of\b", re.IGNORECASE),
    ),
    (
        "in_the_matter_of",
        re.compile(r"\bin the matter of\b", re.IGNORECASE),
    ),
    (
        "order_specific_reference",
        re.compile(
            r"\b(?:in|from)\s+(?:the\s+)?(?:sebi\s+)?order\b",
            re.IGNORECASE,
        ),
    ),
    (
        "in_this_case_or_order",
        re.compile(r"\bin this (?:case|order)\b", re.IGNORECASE),
    ),
)
_HISTORICAL_OFFICIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "previous_chairperson",
        re.compile(
            r"\b(?:previous|former|immediate past)\s+chair(?:man|person)\s+of\s+sebi\b|\bwho\s+was\s+the\s+(?:previous|former)\s+chair(?:man|person)\s+of\s+sebi\b",
            re.IGNORECASE,
        ),
    ),
    (
        "historical_person",
        re.compile(r"^\s*who\s+was\s+[a-z][a-z .'-]{2,80}\s*\??$", re.IGNORECASE),
    ),
    (
        "former_board_member",
        re.compile(
            r"\b(?:previous|former)\s+board members?\s+of\s+sebi\b",
            re.IGNORECASE,
        ),
    ),
)
_GENERAL_EXPLANATORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("what_is", re.compile(r"^\s*what is\b", re.IGNORECASE)),
    ("who_is", re.compile(r"^\s*who is\b", re.IGNORECASE)),
    ("sebi_definition", re.compile(r"^\s*(?:what|who)\s+(?:is|are)\s+sebi\b", re.IGNORECASE)),
    ("explain", re.compile(r"^\s*explain\b", re.IGNORECASE)),
    ("difference_between", re.compile(r"\bdifference between\b", re.IGNORECASE)),
    ("meaning_of", re.compile(r"\bmeaning of\b", re.IGNORECASE)),
    ("tell_me_about", re.compile(r"^\s*tell me about\b", re.IGNORECASE)),
)
_BRIEF_SUMMARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "brief_summary",
        re.compile(r"\bbrief\s+summary\b|\bsummar(?:ise|ize)\s+briefly\b", re.IGNORECASE),
    ),
    (
        "brief_what_happened",
        re.compile(r"\bbrief\s+summary\s+of\s+what\s+happened\b", re.IGNORECASE),
    ),
    (
        "summary_request",
        re.compile(r"\b(?:summar(?:ise|ize)|summary of)\b", re.IGNORECASE),
    ),
)
_NON_SEBI_PERSON_QUERY_RE = re.compile(
    r"^\s*who\s+(?:is|was)\s+[a-z][a-z .'-]{2,120}\s*\??$",
    re.IGNORECASE,
)
_SETTLEMENT_SIGNAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("settlement_order", re.compile(r"\bsettlement\s+order\b", re.IGNORECASE)),
    ("terms_of_settlement", re.compile(r"\bterms?\s+of\s+settlement\b", re.IGNORECASE)),
    ("settlement_application", re.compile(r"\bsettlement\s+applications?\b", re.IGNORECASE)),
    ("matter_settled", re.compile(r"\bmatter\s+settled\b", re.IGNORECASE)),
    ("settlement_proceedings", re.compile(r"\bsettlement\s+proceedings\b", re.IGNORECASE)),
    (
        "settlement_directive",
        re.compile(r"\bwhat\s+did\s+sebi\s+(?:finally\s+)?direct\b", re.IGNORECASE),
    ),
    (
        "settlement_amount",
        re.compile(r"\bwhat\s+was\s+the\s+settlement\s+amount\b", re.IGNORECASE),
    ),
    (
        "finally_ordered_in_settlement",
        re.compile(
            r"\bwhat\s+was\s+finally\s+ordered\s+in\s+the\s+settlement\b",
            re.IGNORECASE,
        ),
    ),
)
_PROCEDURAL_OR_OUTCOME_TERMS: tuple[str, ...] = (
    "dismissed",
    "allowed",
    "upheld",
    "quashed",
    "penalty",
    "penalties",
    "refund",
    "restrained",
    "debarred",
    "directed",
    "direction",
    "directions",
    "operative",
    "finding",
    "findings",
    "settle",
    "settlement",
    "settled",
    "violated",
    "violations",
    "holding",
)
_GENERAL_TOPIC_TERMS: tuple[str, ...] = (
    "settlement order",
    "regulation",
    "regulations",
    "section",
    "rule",
    "rules",
    "sat",
    "nclt",
    "rti act",
    "sebi act",
)
_ORDER_SIGNATORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("signatory", re.compile(r"\b(?:who|which)\b.*\b(?:signed|signatory)\b", re.IGNORECASE)),
    ("signed_order", re.compile(r"\bsigned the order\b", re.IGNORECASE)),
    (
        "quasi_judicial_authority",
        re.compile(
            r"\bwho\s+was\s+the\s+(?:quasi|qasi)[- ]judicial\s+authority\b|\b(?:quasi|qasi)[- ]judicial\s+authority\b",
            re.IGNORECASE,
        ),
    ),
)
_ORDER_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("order_date", re.compile(r"\bwhen\s+was\s+(?:this|the)\s+(?:order|case|matter)\s+(?:passed|issued)\b", re.IGNORECASE)),
    ("passed_date", re.compile(r"\border\s+passed\b", re.IGNORECASE)),
)
_LEGAL_PROVISION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("legal_sections", re.compile(r"\b(?:violation )?sections?\b", re.IGNORECASE)),
    ("legal_regulations", re.compile(r"\bregulations?\b", re.IGNORECASE)),
    ("legal_provisions", re.compile(r"\bprovisions?\b", re.IGNORECASE)),
    ("legal_violations", re.compile(r"\bviolations?\b", re.IGNORECASE)),
)
_LEGAL_EXPLANATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("explain_provisions", re.compile(r"\bexplain\b.*\b(?:sections?|regulations?|provisions?)\b", re.IGNORECASE)),
    ("what_they_mean", re.compile(r"\bwhat they mean\b", re.IGNORECASE)),
    ("explain_them", re.compile(r"\bexplain them\b", re.IGNORECASE)),
)
_ORDER_PAN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pan", re.compile(r"\bpan\b", re.IGNORECASE)),
    ("pan_number", re.compile(r"\bpan\s*(?:number|no\.?)\b", re.IGNORECASE)),
)
_ORDER_AMOUNT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("amount", re.compile(r"\bamount\b", re.IGNORECASE)),
    ("penalty_amount", re.compile(r"\b(?:penalty|settlement|refund|sum)\s+amount\b", re.IGNORECASE)),
    ("rupee_amount", re.compile(r"\b(?:rs\.?|inr|crore|lakh)\b", re.IGNORECASE)),
)
_ORDER_HOLDING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("shares", re.compile(r"\bshares?\b", re.IGNORECASE)),
    ("percentage", re.compile(r"\bpercentage\b|\d+(?:\.\d+)?\s*%", re.IGNORECASE)),
    ("holding", re.compile(r"\bholding(?:s)?\b|\bshareholding\b", re.IGNORECASE)),
    ("ownership", re.compile(r"\bown(?:s|ed|ership)?\b", re.IGNORECASE)),
)
_ORDER_PARTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("parties", re.compile(r"\bpart(?:y|ies)\b", re.IGNORECASE)),
    ("noticees", re.compile(r"\bnoticees?\b", re.IGNORECASE)),
    ("appellants", re.compile(r"\bappellants?\b", re.IGNORECASE)),
    ("respondents", re.compile(r"\brespondents?\b", re.IGNORECASE)),
)
_ORDER_OBSERVATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "da_observation",
        re.compile(
            r"\b(?:da|designated authority|enquiry report|enquiry officer)\b.*\b(?:observe|observed|find|finding|conclude|concluded)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "generic_observations",
        re.compile(
            r"\b(?:what were the observations|what did .* observe|what did .* find|what did .* conclude|what did .* hold)\b",
            re.IGNORECASE,
        ),
    ),
)
_ORDER_NUMERIC_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "price_increase",
        re.compile(
            r"\b(?:how much did .* share price increase|price increased by|share price increase|percentage increase|percent increase)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "price_before_after",
        re.compile(
            r"\b(?:price before and after|before and after the increase|what was the price before and after)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "price_movement",
        re.compile(
            r"\b(?:price movement|price movements|movement for each period|movement of .* for each period|each patch|each period|patch-wise|period-wise)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "listing_high_low",
        re.compile(
            r"\b(?:listing price|listed at|highest price|peak price|lowest price|low price)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ipo_proceeds",
        re.compile(r"\b(?:ipo proceeds|proceeds were raised)\b", re.IGNORECASE),
    ),
)
_FREEFORM_NUMERIC_ORDER_SUBJECT_RE = re.compile(
    r"(?:"
    r"\b(?:price movement|price movements|listing price|listed at|highest price|peak price|lowest price|low price|share price increase|percentage increase|price before and after|before and after the increase)\b.*\b(?:of|for)\s+[a-z0-9][a-z0-9 .&'/-]{2,140}\b"
    r"|"
    r"\bhow much did\s+[a-z0-9][a-z0-9 .&'/-]{2,140}?\s+share price increase\b"
    r"|"
    r"\bwhat was the price before and after(?: the increase)? in\s+[a-z0-9][a-z0-9 .&'/-]{2,140}\b"
    r")",
    re.IGNORECASE,
)
_ACTIVE_MATTER_FOLLOW_UP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "exemption_granted",
        re.compile(
            r"\b(?:what|which)\s+exemption\s+was\s+granted\b|\bwhat\s+relief\s+was\s+granted\b",
            re.IGNORECASE,
        ),
    ),
    (
        "appellate_authority_decision",
        re.compile(
            r"\bwhat\s+did\s+the\s+appellate\s+authority\s+decide\b|\bwhat\s+was\s+the\s+appellate\s+decision\b",
            re.IGNORECASE,
        ),
    ),
    (
        "da_observed",
        re.compile(
            r"\bwhat\s+did\s+the\s+(?:da|designated authority)\s+(?:observe|find|conclude)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "settlement_amount",
        re.compile(r"\bwhat\s+was\s+the\s+settlement\s+amount\b", re.IGNORECASE),
    ),
    (
        "penalty",
        re.compile(
            r"\bwhat\s+was\s+the\s+penalt(?:y|ies)\b|\bwhat\s+penalt(?:y|ies)\s+were\s+imposed\b|\bwhat\s+sentence\s+was\s+imposed\b",
            re.IGNORECASE,
        ),
    ),
    (
        "final_direction",
        re.compile(
            r"\bwhat\s+was\s+the\s+final\s+direction\b|\bwhat\s+did\s+sebi\s+finally\s+direct\b|\bwhat\s+was\s+the\s+action\s+taken\b|\bwhat\s+action\s+(?:was\s+taken|did\s+sebi\s+take)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "outcome",
        re.compile(
            r"\bwhat\s+was\s+the\s+outcome\b|\bwhat\s+happened\b|\bwhat\s+did\s+sebi\s+order\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sat_hold",
        re.compile(
            r"\bwhat\s+did\s+(?:sat|the\s+tribunal|the\s+special\s+court|the\s+court)\s+hold\b|\bwhat\s+did\s+the\s+appellate\s+authority\s+hold\b",
            re.IGNORECASE,
        ),
    ),
)
_SAT_COURT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("vs_sebi", re.compile(r"\b(?:vs\.?|versus|v\.)\s+sebi\b|\bsebi\s+(?:vs\.?|versus|v\.)\b", re.IGNORECASE)),
    ("versus_generic", re.compile(r"\b(?:vs\.?|versus|v\.)\b", re.IGNORECASE)),
    ("sat", re.compile(r"\bsat\b", re.IGNORECASE)),
    ("appeal", re.compile(r"\bappeal\b", re.IGNORECASE)),
    ("wp", re.compile(r"\bw\.?\s*p\.?\b|\bwrit petition\b", re.IGNORECASE)),
    ("court", re.compile(r"\bcourt\b|\bjudgment\b", re.IGNORECASE)),
    ("tribunal", re.compile(r"\btribunal\b", re.IGNORECASE)),
)
_CORPUS_METADATA_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bucket_date_range",
        re.compile(r"\b(?:date range|from which dates|from what dates)\b", re.IGNORECASE),
    ),
    ("bucket_count", re.compile(r"\bhow many\b|\bcount\b", re.IGNORECASE)),
    ("local_pdf_count", re.compile(r"\blocal pdfs?\b|\bpdf availability\b", re.IGNORECASE)),
    ("category_count", re.compile(r"\bhow many categories\b|\bbuckets?\b", re.IGNORECASE)),
)
_MATTER_REFERENCE_TERMS: tuple[str, ...] = (
    "appeal",
    "petition",
    "writ",
    "case",
    "suit",
    "matter",
    "order no",
    "filed by",
    "in the matter of",
    "petitioner",
    "respondent",
)
_SETTLEMENT_EXPLANATORY_OPENERS: tuple[str, ...] = (
    "what is a settlement order",
    "what is settlement order",
    "explain settlement order",
    "explain settlement orders",
    "meaning of settlement order",
    "meaning of settlement proceedings",
)
_GENERIC_ORDER_TOPIC_TERMS: tuple[str, ...] = (
    "settlement order",
    "adjudication order",
    "corrigendum",
    "rti appellate order",
    "exemption order",
    "sat order",
    "ex parte interim order",
    "ex-parte interim order",
    "regulation 30a",
)


def analyze_query(
    query: str,
    *,
    session_state: ChatSessionStateRecord | None = None,
    control_pack: ControlPack | None = None,
) -> QueryAnalysis:
    """Extract deterministic route features from one user query."""

    current_expansion = expand_query(query, contexts=("current_people", "current_offices"))
    order_expansion = expand_query(query, contexts=("order_lookup", "order_legal"))
    normalized_expansions = tuple(
        dict.fromkeys(
            value
            for value in (*current_expansion.expansions, *order_expansion.expansions)
            if value
        )
    )
    pattern_query = " ".join(normalized_expansions) if normalized_expansions else _normalize(query)
    current_info_query = normalize_current_info_query(query, session_state=session_state)
    normalized_query = (
        current_info_query.normalized_query
        if current_info_query.query_family != "unsupported"
        else _normalize(order_expansion.normalized_query or query)
    )
    title_or_party_lookup_signals = _pattern_labels(query, _TITLE_LOOKUP_PATTERNS)
    follow_up_signals = _pattern_labels(pattern_query, _FOLLOW_UP_PATTERNS)
    smalltalk_signals = _pattern_labels(query, _SMALLTALK_PATTERNS)
    structured_current_info_signals = tuple(
        dict.fromkeys(
            ()
            if current_info_query.query_family == "unsupported"
            else (current_info_query.query_family,)
            + (("office_follow_up",) if current_info_query.is_follow_up else ())
        )
    )
    current_official_lookup_signals = _pattern_labels(query, _CURRENT_OFFICIAL_LOOKUP_PATTERNS)
    current_news_signals = _pattern_labels(query, _CURRENT_NEWS_PATTERNS)
    historical_official_signals = _pattern_labels(query, _HISTORICAL_OFFICIAL_PATTERNS)
    current_public_fact_signals = _pattern_labels(query, _CURRENT_PUBLIC_FACT_PATTERNS)
    company_role_query = parse_company_role_query(query)
    company_role_signals = (
        company_role_query.matched_signals
        if company_role_query is not None
        else _pattern_labels(query, _COMPANY_ROLE_PATTERNS)
    )
    order_context_override_signals = tuple(
        dict.fromkeys(
            (
                *_pattern_labels(query, _ORDER_CONTEXT_OVERRIDE_PATTERNS),
                *detect_company_role_order_context(query),
            )
        )
    )
    brief_summary_signals = _pattern_labels(query, _BRIEF_SUMMARY_PATTERNS)
    procedural_or_outcome_signals = _matched_terms(
        pattern_query,
        _PROCEDURAL_OR_OUTCOME_TERMS,
    )
    settlement_signals = _settlement_signals(query, pattern_query)
    general_explanatory_signals = _general_signals(query, pattern_query)
    matter_reference_signals = _matter_reference_signals(query, order_expansion.normalized_query)
    order_signatory_signals = _pattern_labels(pattern_query, _ORDER_SIGNATORY_PATTERNS)
    order_date_signals = _pattern_labels(pattern_query, _ORDER_DATE_PATTERNS)
    legal_provision_signals = _pattern_labels(pattern_query, _LEGAL_PROVISION_PATTERNS)
    legal_explanation_signals = _pattern_labels(pattern_query, _LEGAL_EXPLANATION_PATTERNS)
    order_pan_signals = _pattern_labels(pattern_query, _ORDER_PAN_PATTERNS)
    order_amount_signals = _pattern_labels(pattern_query, _ORDER_AMOUNT_PATTERNS)
    order_holding_signals = _pattern_labels(pattern_query, _ORDER_HOLDING_PATTERNS)
    order_party_signals = _pattern_labels(pattern_query, _ORDER_PARTY_PATTERNS)
    order_observation_signals = _pattern_labels(pattern_query, _ORDER_OBSERVATION_PATTERNS)
    order_numeric_fact_signals = _pattern_labels(pattern_query, _ORDER_NUMERIC_FACT_PATTERNS)
    sat_court_signals = _pattern_labels(query, _SAT_COURT_PATTERNS)
    corpus_metadata_signals = _pattern_labels(query, _CORPUS_METADATA_PATTERNS)
    general_topic_query = _looks_like_generic_order_topic_query(query, pattern_query)
    mentions_sebi = "sebi" in pattern_query or "securities and exchange board of india" in pattern_query
    company_role_current_fact = bool(company_role_query) and not (
        mentions_sebi
        or order_context_override_signals
        or bool(company_role_query and company_role_query.explicit_order_context)
    )
    comparison_terms = detect_comparison_terms(query)
    strict_matter_lock = (
        StrictMatterLock(
            comparison_intent=bool(comparison_terms),
            comparison_terms=comparison_terms,
            reason_codes=("company_role_current_fact",),
        )
        if (
            general_topic_query
            or "sebi_definition" in general_explanatory_signals
            or company_role_current_fact
        )
        else resolve_strict_matter_lock(
            query=query,
            control_pack=control_pack,
            title_lookup_signals=title_or_party_lookup_signals,
            matter_reference_signals=matter_reference_signals,
        )
    )

    has_active_documents = bool(session_state and session_state.active_document_ids)
    has_active_record_keys = bool(session_state and session_state.active_record_keys)
    has_session_scope = has_active_documents or has_active_record_keys
    has_active_clarification = bool(session_state and session_state.clarification_context)
    likely_follow_up = bool(follow_up_signals) and has_session_scope
    asks_order_signatory = bool(order_signatory_signals)
    asks_order_date = bool(order_date_signals)
    asks_legal_provisions = bool(legal_provision_signals)
    asks_provision_explanation = bool(legal_explanation_signals) or (
        asks_legal_provisions and "explain" in pattern_query
    )
    asks_order_pan = bool(order_pan_signals)
    asks_order_amount = bool(order_amount_signals) and (
        bool(title_or_party_lookup_signals)
        or bool(matter_reference_signals)
        or bool(strict_matter_lock.named_matter_query)
        or has_session_scope
        or any(token in pattern_query for token in ("order", "case", "matter", "appeal"))
    )
    asks_order_holding = bool(order_holding_signals) and (
        bool(title_or_party_lookup_signals)
        or bool(matter_reference_signals)
        or bool(strict_matter_lock.named_matter_query)
        or has_session_scope
        or any(
            token in pattern_query
            for token in ("limited", "ltd", "trust", "company", "acquirer", "proposed")
        )
    )
    asks_order_parties = bool(order_party_signals) and (
        bool(title_or_party_lookup_signals)
        or bool(matter_reference_signals)
        or bool(strict_matter_lock.named_matter_query)
        or has_session_scope
    )
    asks_order_observations = bool(order_observation_signals) or (
        has_session_scope
        and any(
            token in pattern_query
            for token in (" observation ", " observations ", " observe ", " observed ", " finding ", " findings ", " conclude ", " concluded ")
        )
    )
    asks_order_numeric_fact = bool(order_numeric_fact_signals) and (
        bool(title_or_party_lookup_signals)
        or bool(matter_reference_signals)
        or bool(strict_matter_lock.named_matter_query)
        or has_session_scope
        or bool(_FREEFORM_NUMERIC_ORDER_SUBJECT_RE.search(query))
    )
    active_matter_follow_up_intent = _resolve_active_matter_follow_up_intent(
        pattern_query=pattern_query,
        has_active_scope=has_session_scope,
    )
    generic_legal_definition = bool(general_explanatory_signals) and not (
        title_or_party_lookup_signals
        or matter_reference_signals
        or strict_matter_lock.named_matter_query
        or _contains_deictic_reference(pattern_query)
    )
    deictic_detail_or_summary_follow_up = bool(
        has_session_scope and _is_active_matter_deictic_detail_follow_up(pattern_query)
    )

    if has_session_scope and not _looks_like_new_lookup(query):
        if (
            procedural_or_outcome_signals
            or _contains_deictic_reference(pattern_query)
            or deictic_detail_or_summary_follow_up
            or asks_order_signatory
            or asks_order_date
            or (asks_legal_provisions and not generic_legal_definition)
            or asks_order_pan
            or asks_order_amount
            or asks_order_holding
            or asks_order_parties
            or asks_order_observations
            or asks_order_numeric_fact
            or active_matter_follow_up_intent is not None
            or (asks_provision_explanation and not generic_legal_definition)
        ):
            likely_follow_up = True

    active_scope_matches_named_matter = bool(
        session_state
        and strict_matter_lock.locked_record_keys
        and set(strict_matter_lock.locked_record_keys).issubset(
            set(session_state.active_record_keys or ())
        )
    )
    fresh_query_override = bool(
        has_session_scope
        and not active_scope_matches_named_matter
        and strict_matter_lock.strict_scope_required
        and not _contains_deictic_reference(pattern_query)
        and (
            _looks_like_new_lookup(query)
            or _has_explicit_named_matter_override(
                query=query,
                normalized_query=pattern_query,
                strict_matter_lock=strict_matter_lock,
                title_lookup_signals=title_or_party_lookup_signals,
                matter_reference_signals=matter_reference_signals,
            )
        )
    )

    if deictic_detail_or_summary_follow_up and not fresh_query_override:
        strict_matter_lock = StrictMatterLock(
            named_matter_query=False,
            strict_scope_required=False,
            strict_single_matter=False,
            ambiguous=False,
            comparison_intent=strict_matter_lock.comparison_intent,
            comparison_terms=strict_matter_lock.comparison_terms,
            matched_aliases=strict_matter_lock.matched_aliases,
            matched_entities=strict_matter_lock.matched_entities,
            matched_titles=strict_matter_lock.matched_titles,
            locked_record_keys=(),
            candidates=(),
            reason_codes=("active_matter_deictic_follow_up",),
        )

    settlement_explanatory = _is_generic_settlement_explanation(
        pattern_query,
        settlement_signals=settlement_signals,
    )
    appears_smalltalk = bool(smalltalk_signals)
    appears_current_news_lookup = bool(current_news_signals)
    appears_sat_court_style = bool(sat_court_signals) and not bool(corpus_metadata_signals)
    appears_corpus_metadata_query = bool(corpus_metadata_signals) and any(
        alias in pattern_query
        for alias in (
            "sat",
            "court",
            "special court",
            "settlement",
            "regulation 30a",
            "30a",
            "categories",
            "buckets",
            "local pdf",
        )
    )
    appears_historical_official_lookup = bool(historical_official_signals) and not (
        title_or_party_lookup_signals or matter_reference_signals
    )
    appears_non_sebi_person_query = bool(
        _NON_SEBI_PERSON_QUERY_RE.search(query)
        and not mentions_sebi
        and not appears_current_news_lookup
        and not appears_historical_official_lookup
        and not title_or_party_lookup_signals
        and not matter_reference_signals
        and not strict_matter_lock.named_matter_query
    )
    requires_live_information = bool(
        company_role_current_fact
        or (
            current_public_fact_signals
            and not title_or_party_lookup_signals
            and not matter_reference_signals
            and not strict_matter_lock.named_matter_query
        )
    )
    appears_structured_current_info = bool(structured_current_info_signals) and not (
        appears_current_news_lookup
        or appears_historical_official_lookup
        or appears_corpus_metadata_query
        or appears_sat_court_style
        or (
            current_info_query.query_family == "person_lookup"
            and not mentions_sebi
            and not (
                _explicit_current_person_query(query)
                or bool(current_info_query.department_hint)
            )
        )
        or (
            strict_matter_lock.named_matter_query
            and current_info_query.query_family == "person_lookup"
            and not (
                _explicit_current_person_query(query)
                or bool(current_info_query.department_hint)
            )
        )
        or (
            (
                asks_order_signatory
                or asks_order_date
                or asks_legal_provisions
                or asks_provision_explanation
                or asks_order_pan
                or asks_order_amount
                or asks_order_holding
                or asks_order_parties
                or asks_order_numeric_fact
            )
            and (
                strict_matter_lock.named_matter_query
                or title_or_party_lookup_signals
                or matter_reference_signals
            )
        )
    )
    appears_current_official_lookup = bool(current_official_lookup_signals) and not (
        appears_current_news_lookup
        or appears_historical_official_lookup
        or appears_structured_current_info
        or appears_corpus_metadata_query
        or appears_sat_court_style
        or (
            strict_matter_lock.named_matter_query
            and current_info_query.query_family == "person_lookup"
            and not _explicit_current_person_query(query)
        )
        or (
            (
                asks_order_signatory
                or asks_order_date
                or asks_legal_provisions
                or asks_provision_explanation
                or asks_order_pan
                or asks_order_amount
                or asks_order_holding
                or asks_order_parties
                or asks_order_numeric_fact
            )
            and (
                strict_matter_lock.named_matter_query
                or title_or_party_lookup_signals
                or matter_reference_signals
            )
        )
    )
    active_order_override = bool(
        has_session_scope
        and not fresh_query_override
        and (
            asks_order_signatory
            or asks_order_date
            or (asks_legal_provisions and not generic_legal_definition)
            or (asks_provision_explanation and not generic_legal_definition)
            or asks_order_pan
            or asks_order_amount
            or asks_order_holding
            or asks_order_parties
            or asks_order_observations
            or asks_order_numeric_fact
            or active_matter_follow_up_intent is not None
            or deictic_detail_or_summary_follow_up
            or (
                _contains_deictic_reference(pattern_query)
                and any(
                    token in pattern_query
                    for token in (" observe ", " observed ", " observation ", " noted ", " note ")
                )
            )
        )
    )
    if (
        likely_follow_up
        and strict_matter_lock.strict_scope_required
        and strict_matter_lock.strict_single_matter
        and strict_matter_lock.locked_record_keys
        and not active_order_override
        and not _contains_deictic_reference(pattern_query)
    ):
        likely_follow_up = False
    if fresh_query_override:
        likely_follow_up = False
    if generic_legal_definition:
        likely_follow_up = False
    appears_settlement_specific = bool(settlement_signals) and not settlement_explanatory and bool(
        title_or_party_lookup_signals
        or matter_reference_signals
        or procedural_or_outcome_signals
        or likely_follow_up
        or _has_named_party_reference(query)
    )
    appears_general_explanatory = bool(general_explanatory_signals) and not (
        appears_smalltalk
        or appears_corpus_metadata_query
        or appears_structured_current_info
        or appears_current_official_lookup
        or title_or_party_lookup_signals
        or matter_reference_signals
        or likely_follow_up
        or appears_settlement_specific
        or company_role_current_fact
    )
    appears_matter_specific = bool(
        title_or_party_lookup_signals
        or matter_reference_signals
        or (procedural_or_outcome_signals and (has_session_scope or _YEAR_RE.search(query)))
        or likely_follow_up
        or appears_settlement_specific
        or strict_matter_lock.named_matter_query
    ) and not (
        appears_smalltalk
        or appears_current_official_lookup
        or appears_corpus_metadata_query
        or company_role_current_fact
    )
    query_family = _resolve_query_family(
        raw_query=query,
        current_info_query=current_info_query,
        strict_matter_lock=strict_matter_lock,
        active_order_override=active_order_override,
        asks_provision_explanation=asks_provision_explanation,
        asks_order_signatory=asks_order_signatory,
        asks_order_date=asks_order_date,
        asks_legal_provisions=asks_legal_provisions,
        asks_order_pan=asks_order_pan,
        asks_order_amount=asks_order_amount,
        asks_order_holding=asks_order_holding,
        asks_order_parties=asks_order_parties,
        asks_order_observations=asks_order_observations,
        asks_order_numeric_fact=asks_order_numeric_fact,
        active_matter_follow_up_intent=active_matter_follow_up_intent,
        appears_current_news_lookup=appears_current_news_lookup,
        appears_historical_official_lookup=appears_historical_official_lookup,
        appears_structured_current_info=appears_structured_current_info,
        appears_current_official_lookup=appears_current_official_lookup,
        appears_company_role_current_fact=company_role_current_fact,
        appears_general_explanatory=appears_general_explanatory,
        appears_matter_specific=appears_matter_specific,
        appears_corpus_metadata_query=appears_corpus_metadata_query,
    )

    return QueryAnalysis(
        raw_query=query,
        normalized_query=normalized_query,
        query_family=query_family,
        normalized_expansions=normalized_expansions,
        matched_abbreviations=tuple(
            dict.fromkeys(
                (*current_expansion.matched_abbreviations, *order_expansion.matched_abbreviations)
            )
        ),
        title_or_party_lookup_signals=title_or_party_lookup_signals,
        procedural_or_outcome_signals=procedural_or_outcome_signals,
        settlement_signals=settlement_signals,
        general_explanatory_signals=general_explanatory_signals,
        smalltalk_signals=smalltalk_signals,
        structured_current_info_signals=structured_current_info_signals,
        current_official_lookup_signals=current_official_lookup_signals,
        current_news_signals=current_news_signals,
        historical_official_signals=historical_official_signals,
        current_public_fact_signals=current_public_fact_signals,
        company_role_signals=company_role_signals,
        order_context_override_signals=order_context_override_signals,
        brief_summary_signals=brief_summary_signals,
        current_info_query_family=(
            None if current_info_query.query_family == "unsupported" else current_info_query.query_family
        ),
        current_info_follow_up=current_info_query.is_follow_up,
        follow_up_signals=follow_up_signals,
        matter_reference_signals=matter_reference_signals,
        sat_court_signals=sat_court_signals,
        corpus_metadata_signals=corpus_metadata_signals,
        asks_order_signatory=asks_order_signatory,
        asks_order_date=asks_order_date,
        asks_legal_provisions=asks_legal_provisions,
        asks_provision_explanation=asks_provision_explanation,
        asks_order_pan=asks_order_pan,
        asks_order_amount=asks_order_amount,
        asks_order_holding=asks_order_holding,
        asks_order_parties=asks_order_parties,
        asks_order_observations=asks_order_observations,
        asks_order_numeric_fact=asks_order_numeric_fact,
        active_matter_follow_up_intent=active_matter_follow_up_intent,
        active_order_override=active_order_override,
        fresh_query_override=fresh_query_override,
        likely_follow_up=likely_follow_up,
        has_active_documents=has_active_documents,
        has_active_record_keys=has_active_record_keys,
        has_session_scope=has_session_scope,
        has_active_clarification=has_active_clarification,
        mentions_sebi=mentions_sebi,
        appears_smalltalk=appears_smalltalk,
        appears_structured_current_info=appears_structured_current_info,
        appears_current_official_lookup=appears_current_official_lookup,
        appears_current_news_lookup=appears_current_news_lookup,
        appears_historical_official_lookup=appears_historical_official_lookup,
        appears_corpus_metadata_query=appears_corpus_metadata_query,
        appears_sat_court_style=appears_sat_court_style,
        appears_non_sebi_person_query=appears_non_sebi_person_query,
        appears_company_role_current_fact=company_role_current_fact,
        appears_general_explanatory=appears_general_explanatory,
        appears_matter_specific=appears_matter_specific,
        appears_settlement_specific=appears_settlement_specific,
        asks_brief_summary=bool(brief_summary_signals),
        requires_live_information=requires_live_information,
        comparison_intent=strict_matter_lock.comparison_intent,
        comparison_terms=strict_matter_lock.comparison_terms,
        strict_scope_required=strict_matter_lock.strict_scope_required,
        strict_single_matter=strict_matter_lock.strict_single_matter,
        strict_lock_record_keys=strict_matter_lock.locked_record_keys,
        strict_lock_titles=tuple(candidate.title for candidate in strict_matter_lock.candidates[:2]),
        strict_lock_matched_aliases=strict_matter_lock.matched_aliases,
        strict_lock_matched_entities=strict_matter_lock.matched_entities,
        strict_lock_reason_codes=strict_matter_lock.reason_codes,
        strict_lock_ambiguous=strict_matter_lock.ambiguous,
        strict_matter_lock=strict_matter_lock,
    )


def _normalize(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", query.strip().lower())


def _pattern_labels(
    query: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, ...]:
    matches = [label for label, pattern in patterns if pattern.search(query)]
    return tuple(matches)


def _matched_terms(normalized_query: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    matches = [term for term in terms if re.search(rf"\b{re.escape(term)}\b", normalized_query)]
    return tuple(matches)


def _general_signals(query: str, normalized_query: str) -> tuple[str, ...]:
    matches = [label for label, pattern in _GENERAL_EXPLANATORY_PATTERNS if pattern.search(query)]
    if _looks_like_topic_fragment(normalized_query):
        matches.append("topic_fragment")
    topic_matches = _matched_terms(normalized_query, _GENERAL_TOPIC_TERMS)
    if matches and topic_matches:
        return tuple(matches) + topic_matches
    if matches and not _YEAR_RE.search(query):
        return tuple(matches)
    if topic_matches and normalized_query.startswith(("what is", "explain", "difference")):
        return topic_matches
    return ()


def _matter_reference_signals(query: str, normalized_query: str) -> tuple[str, ...]:
    matches = list(_matched_terms(normalized_query, _MATTER_REFERENCE_TERMS))
    if _YEAR_RE.search(query):
        matches.append("year")
    if _PROPER_NAME_RE.search(query) and any(
        term in normalized_query for term in ("appeal", "matter", "filed by", "vs", "versus")
    ):
        matches.append("proper_name")
    if not normalized_query.endswith("?") and _looks_like_new_lookup(query):
        matches.append("title_like")
    return tuple(dict.fromkeys(matches))


def _settlement_signals(query: str, normalized_query: str) -> tuple[str, ...]:
    matches = list(_pattern_labels(query, _SETTLEMENT_SIGNAL_PATTERNS))
    matches.extend(
        term
        for term in ("settlement", "settled")
        if re.search(rf"\b{term}\b", normalized_query)
    )
    return tuple(dict.fromkeys(matches))


def _has_explicit_named_matter_override(
    *,
    query: str,
    normalized_query: str,
    strict_matter_lock: StrictMatterLock,
    title_lookup_signals: tuple[str, ...],
    matter_reference_signals: tuple[str, ...],
) -> bool:
    if not strict_matter_lock.strict_scope_required:
        return False
    if not strict_matter_lock.locked_record_keys and not strict_matter_lock.candidates:
        return False
    if len(strict_matter_lock.candidates) == 1:
        candidate = strict_matter_lock.candidates[0]
        score = float(getattr(candidate, "score", 0.0) or 0.0)
        overlap = float(getattr(candidate, "title_overlap_ratio", 0.0) or 0.0)
        if score >= 0.44 and (overlap >= 0.14 or _has_named_party_reference(query)):
            return True
    if title_lookup_signals or matter_reference_signals:
        return True
    if any(token in normalized_query for token in ("case", "matter", "order", "appeal", "petition")):
        return bool(_has_named_party_reference(query) or strict_matter_lock.matched_entities)
    return bool(_has_named_party_reference(query) and strict_matter_lock.matched_entities)


def _looks_like_new_lookup(query: str) -> bool:
    stripped = query.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return False
    return bool(_YEAR_RE.search(stripped) or any(
        marker in stripped.lower()
        for marker in ("filed by", "in the matter of", "appeal no", "petition no", "vs", "versus")
    ))


def _contains_deictic_reference(normalized_query: str) -> bool:
    return any(
        phrase in normalized_query
        for phrase in (
            "this case",
            "this matter",
            "this appeal",
            "this order",
            "that case",
            "that matter",
            "the order",
            "these sections",
            "these regulations",
            "these provisions",
        )
    )


def _is_active_matter_deictic_detail_follow_up(normalized_query: str) -> bool:
    return bool(
        _contains_deictic_reference(normalized_query)
        and _DEICTIC_DETAIL_OR_SUMMARY_RE.search(normalized_query)
    )


def _is_generic_settlement_explanation(
    normalized_query: str,
    *,
    settlement_signals: tuple[str, ...],
) -> bool:
    if not settlement_signals:
        return False
    if normalized_query.startswith(("what is", "explain", "meaning of")):
        return True
    return any(normalized_query.startswith(prefix) for prefix in _SETTLEMENT_EXPLANATORY_OPENERS)


def _has_named_party_reference(query: str) -> bool:
    tokens = [
        token
        for token in _CAPITALIZED_TOKEN_RE.findall(query)
        if token.upper() not in {"SEBI", "WHAT", "SETTLEMENT", "ORDER"}
    ]
    return len(tokens) >= 2


def _looks_like_topic_fragment(normalized_query: str) -> bool:
    if not normalized_query:
        return False
    if normalized_query.startswith(("on ", "about ")):
        return len(normalized_query.split()) <= 4
    return False


def _looks_like_generic_order_topic_query(query: str, normalized_query: str) -> bool:
    if _YEAR_RE.search(query):
        return False
    if any(
        marker in normalized_query
        for marker in ("filed by", "in the matter of", "appeal no", "petition no", " vs ", " versus ")
    ):
        return False
    if not normalized_query.startswith(("what is", "what is an", "what is a", "explain")):
        return False
    return any(term in normalized_query for term in _GENERIC_ORDER_TOPIC_TERMS)


def _resolve_active_matter_follow_up_intent(
    *,
    pattern_query: str,
    has_active_scope: bool,
) -> str | None:
    if not has_active_scope:
        return None
    for label, pattern in _ACTIVE_MATTER_FOLLOW_UP_PATTERNS:
        if pattern.search(pattern_query):
            return label
    return None


def _explicit_current_person_query(query: str) -> bool:
    normalized_query = _normalize(query)
    person_tail = ""
    if normalized_query.startswith("who is "):
        person_tail = normalized_query[len("who is ") :].strip(" ?")
    if person_tail and len(person_tail.split()) == 1 and person_tail.isalpha():
        return True
    if normalized_query and len(normalized_query.split()) == 1 and normalized_query.isalpha() and len(normalized_query) >= 4:
        return True
    return any(
        phrase in normalized_query
        for phrase in (
            "designation of",
            "join",
            "phone",
            "number",
            "staff id",
            "staff number",
            "staff no",
            "email",
            " in sebi",
            " at sebi",
            " from sebi",
            " sebi",
            "called ",
        )
    )


def _resolve_query_family(
    *,
    raw_query: str,
    current_info_query,
    strict_matter_lock: StrictMatterLock,
    active_order_override: bool,
    asks_provision_explanation: bool,
    asks_order_signatory: bool,
    asks_order_date: bool,
    asks_legal_provisions: bool,
    asks_order_pan: bool,
    asks_order_amount: bool,
    asks_order_holding: bool,
    asks_order_parties: bool,
    asks_order_observations: bool,
    asks_order_numeric_fact: bool,
    active_matter_follow_up_intent: str | None,
    appears_current_news_lookup: bool,
    appears_historical_official_lookup: bool,
    appears_structured_current_info: bool,
    appears_current_official_lookup: bool,
    appears_company_role_current_fact: bool,
    appears_general_explanatory: bool,
    appears_matter_specific: bool,
    appears_corpus_metadata_query: bool,
) -> str:
    if active_order_override:
        if active_matter_follow_up_intent:
            return f"active_matter_{active_matter_follow_up_intent}_follow_up"
        if asks_provision_explanation:
            return "active_order_legal_explanation"
        if asks_order_signatory:
            return "active_order_signatory_follow_up"
        if asks_order_date:
            return "active_order_date_follow_up"
        if asks_legal_provisions:
            return "active_order_legal_follow_up"
        if asks_order_pan:
            return "active_order_pan_follow_up"
        if asks_order_amount:
            return "active_order_amount_follow_up"
        if asks_order_holding:
            return "active_order_holding_follow_up"
        if asks_order_parties:
            return "active_order_party_follow_up"
        if asks_order_observations:
            return "active_order_observation_follow_up"
        if asks_order_numeric_fact:
            return "active_order_numeric_fact_follow_up"
        return "active_order_follow_up"
    if appears_corpus_metadata_query:
        return "corpus_metadata_query"
    if appears_current_news_lookup:
        return "current_news_query"
    if appears_historical_official_lookup:
        return "historical_official_query"
    if appears_current_official_lookup:
        return "current_official_query"
    if appears_company_role_current_fact:
        return "general_knowledge"
    if strict_matter_lock.named_matter_query and appears_matter_specific:
        return "named_order_exact_entity" if _is_terse_named_entity_query(raw_query) else "named_order_query"
    if appears_structured_current_info and current_info_query.query_family != "unsupported":
        if current_info_query.query_family in {"person_lookup", "staff_id_lookup"}:
            return "structured_people_query"
        if current_info_query.query_family in {"office_contact", "regional_director"}:
            return "structured_office_query"
        if current_info_query.query_family in {"designation_count", "wtm_list", "ed_list", "leadership_list", "board_members", "total_strength"}:
            return "structured_aggregate_query"
        return "structured_current_info"
    if appears_general_explanatory:
        return "general_knowledge"
    if appears_matter_specific:
        return "matter_specific"
    return "ambiguous"


def _is_terse_named_entity_query(query: str) -> bool:
    normalized_query = _normalize(query)
    tokens = normalized_query.split()
    if not tokens or len(tokens) > 6:
        return False
    return tokens[0] not in {
        "tell",
        "what",
        "which",
        "who",
        "how",
        "why",
        "when",
        "explain",
        "summary",
        "compare",
    }
