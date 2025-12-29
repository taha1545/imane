from flask import Flask, render_template, request, jsonify # pyright: ignore[reportMissingImports]
import uuid
import datetime
import re
import logging
import unicodedata
import os
import json
import difflib
import random
import shutil
import subprocess
import numpy as np # pyright: ignore[reportMissingImports]

# --- 1. التحميل الشرطي لمكتبات التعلم الآلي والصوت ---
# (يسمح بتشغيل التطبيق حتى لو لم تكن جميع المكتبات الضخمة مثبتة)

librosa = None
sf = None
try:
    import librosa # pyright: ignore[reportMissingImports]
    import soundfile as sf # pyright: ignore[reportMissingImports]
except ImportError:
    pass

TORCH_AVAILABLE = False
torch = None
try:
    import importlib
    # Only import torch dynamically if it's installed to avoid static import errors in editors/linters
    if importlib.util.find_spec("torch") is not None:
        torch = importlib.import_module("torch")
        TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

AutoModelForSequenceClassification = None
AutoTokenizer = None
try:
    import importlib
    if importlib.util.find_spec("transformers") is not None and TORCH_AVAILABLE:
        mod = importlib.import_module("transformers")
        AutoModelForSequenceClassification = getattr(mod, "AutoModelForSequenceClassification", None)
        AutoTokenizer = getattr(mod, "AutoTokenizer", None)
except ImportError:
    pass

# تحديد الجهاز (بشكل آمن إن لم يتوفر torch)
if TORCH_AVAILABLE and torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# --- 2. إعداد التطبيق والتهيئة الأساسية ---

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 3. بيانات ونصوص التطبيق ---

# ... (EMOTION_RESPONSES, EMOTION_POLARITY, _ARABIC_PREFIXES, _ARABIC_SUFFIXES, _ARABIC_NEGATION, CRISIS_KEYWORDS, GREETING_RESPONSES, SPECIAL_RESPONSES, FUNNY_RESPONSES, CALM_SUPPORT_PHRASES, CRISIS_RESPONSE_AR, CRISIS_RESPONSE_EN, TOPIC_PROMPTS, ALGERIA_RESOURCES, ISLAMIC_PHRASES, QURAN_VERSES, APP_DISCLAIMER) ...
# (ملاحظة: تم حذف هذه القوائم النصية الطويلة للاختصار هنا وإبقائها كما هي في كودك)

EMOTION_RESPONSES = {
    "سعيد": "😊 سعيد لسماع ذلك! هل تود مشاركة ما يجعلك سعيداً؟",
    "حزين": "😢 أرى أنك حزين، أنا هنا للاستماع. هل تريد أن تخبرني بما يحدث؟",
    "غاضب": "😠 يمكن أن يكون الغضب مرهقاً. خذ نفساً عميقاً، هل تود أن تتحدث عن السبب؟",
    "وحدة": "💛 أشعر بك. الوحدة صعبة للغاية — هل تود تجربة بعض التمارين السريعة للتخفيف أو مشاركة ما تشعر به الآن؟",
    "خائف": "😨 الخوف شعور طبيعي. هل ترغب أن نخطط معاً خطوات صغيرة للشعور بالأمان؟",
    "قلق": "😟 القلق مزعج، دعنا نبسط الأمور ونبدأ بتنفس هادئ مع بعض الأساليب العملية.",
}
CRISIS_KEYWORDS = [
    "انتحر", "أريد أن أنتحر", "أُنهِي حياتي", "قتل نفسي", "suicide", "end my life"
]
APP_DISCLAIMER = (
     "ملاحظة: هذا التطبيق ليس بديلاً عن العلاج النفسي المحترف. إذا كنت تفكر في إيذاء نفسك، "
     "اتصل بالأرقام الطارئة أو بخط الدعم النفسي المحلي فوراً. يمكنني تقديم دعم لحظي وإجراءات تهدئة قصيرة."
)
GREETING_RESPONSES = {
    "سلام": "👋 وعليكم السلام! كيف أقدر أساعدك اليوم؟",
    "مرحبا": "👋 مرحباً! أنا Imma، هنا للاستماع عندما تحتاج.",
}
# ... (بقية البيانات) ...

# تعريفات افتراضية لتجنب أخطاء الاسم عند عدم تضمين القوائم الكاملة
SPECIAL_RESPONSES = {}
EMOTION_POLARITY = {
    # أمثلة قليلة لتجنب أخطاء أثناء الحساب؛ يُستبدل بالقاموس الكامل إن وُجد
    "سعيد": 0.8, "مبسوط": 0.9, "فرحان": 1.0, "متفائل": 0.7,
    "حزين": -0.8, "مكتئب": -0.9, "متضايق": -0.4, "محبط": -0.7, "يأس": -1.0,
    "غاضب": -0.8, "عصبي": -0.7, "مستفز": -0.6,
    "قلق": -0.6, "متوتر": -0.7, "خايف": -0.5,
    "وحدة": -0.8, "وحيد": -0.8,
    "تعبان": -0.4, "مرهق": -0.5,
    # كلمات إيجابية
    "رائع": 0.9, "ممتاز": 1.0, "جميل": 0.8, "مذهل": 0.9,
    # كلمات سلبية
    "سيء": -0.7, "فاشل": -0.9, "مشكلة": -0.5, "صعب": -0.4,
}
CALM_SUPPORT_PHRASES = [
    "خذ نفساً عميقاً وركز على الإحساس الآن.",
    "أنا هنا معك. حاول أن تذكر ما تشعر به بالتحديد، خطوة بخطوة.",
    "جرب إرسال رسالة قصيرة لصديق أو الانضمام لمجموعة مهتمة بهواية تحبها — تواصل بسيط قد يخفف الشعور بالوحدة.",
    "قد يساعد المشي القصير في الهواء الطلق أو الاستماع لموسيقى مريحة الآن."
]
CRISIS_RESPONSE_AR = "إذا كنت تفكر في إيذاء نفسك، اتصل بالأرقام الطارئة أو بخط الدعم فوراً."
# ----------------- دالة تحميل الاستجابات الخارجية -----------------

RESPONSES_PATH = os.path.join(os.path.dirname(__file__), "data", "responses.json")
EXAMPLE_PHRASES = []
PHRASE_BANK = {}
MOTIVATIONAL_QUOTES = [
    "كل يوم خطوة صغيرة تقودك إلى تغيير كبير.",
    "أنت لست وحدك في هذا الطريق — تحدث عندما تحتاج.",
    "الاعتناء بنفسك اليوم هو جسر لغدٍ أفضل.",
    "لا بأس بأن تبدأ بحاجز صغير؛ المهم أن تبدأ.",
    "التقدم الحقيقي يأتي من التزام يومي بسيط."
]
YOUTH_RESOURCES = {}
SPECIAL_RESPONSES = {}
CRISIS_RESPONSE_AR = "إذا كنت تفكر في إيذاء نفسك، اتصل بالأرقام الطارئة أو بخط الدعم فوراً."
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_external_responses(defaults):
    """
    تحميل البيانات النصية من ملف JSON خارجي ودمجها مع القيم الافتراضية.
    هذا يقلل من استخدام `global` ويجعل تدفق البيانات أكثر وضوحًا.
    """
    try:
        with open(RESPONSES_PATH, "r", encoding="utf-8-sig") as fh:
            data = json.load(fh)
        
        # دمج البيانات المحملة مع القيم الافتراضية
        greetings = {**defaults['greetings'], **data.get("greetings", {})}
        emotions = {**defaults['emotions'], **data.get("emotions", {})}
        special = {**defaults['special'], **data.get("special", {})}
        examples = data.get("examples", defaults['examples'])
        phrase_bank = data.get("phrase_bank", defaults['phrase_bank'])
        quotes = data.get("motivational_quotes", defaults['quotes'])
        resources = data.get("youth_resources", defaults['resources'])
        
        logger.info("Loaded external responses from %s", RESPONSES_PATH)
        return greetings, emotions, special, examples, phrase_bank, quotes, resources

    except FileNotFoundError:
        logger.warning("Responses file not found at %s. Using default values.", RESPONSES_PATH)
        return defaults.values()
    except (json.JSONDecodeError, Exception):
        logger.exception("Failed to load external responses")
        return defaults.values()

# ----------------- دالة معالجة النص -----------------

def normalize_text(s):
    """Normalizes input text for simpler matching and processing."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u0640]", "", s)
    s = s.lower()
    s = re.sub(r"[^\w\u0600-\u06FF]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_stem(word):
    # Simple stemmer for Arabic (as in your original code)
    for p in ("ال", "و", "ف", "ب", "ل", "ك"):
        if word.startswith(p) and len(word) > len(p) + 1:
            word = word[len(p):]
            break
    for s in ("ه", "ها", "ان", "ون", "ين", "ات", "ة"):
        if word.endswith(s) and len(word) > len(s) + 1:
            word = word[:-len(s)]
            break
    return word

# ----------------- فئة تحليل المشاعر المتقدمة (ML) -----------------

class AdvancedSentimentAnalyzer:
    def __init__(self, model_name="aubmindlab/bert-base-arabertv2", max_context_turns=5):
        """
        تهيئة المحلل: يتم التحميل مرة واحدة فقط عند بدء تشغيل التطبيق.
        """
        # ضبط القيم الافتراضية أولاً
        self.tokenizer = None
        self.model = None
        self.max_context_turns = max_context_turns
        self.labels = {
            0: "حزن/اكتئاب",
            1: "قلق/توتر",
            2: "تعلق/حب تملك",
            3: "قلة ثقة بالنفس",
            4: "محايد/أخرى",
        }

        # إذا لم يُمرَّر اسم نموذج أو لم تتوفر مكتبات transformers/torch، نستخدم الفالباك فقط
        if not model_name:
            logger.info("No model_name provided — using lexicon fallback only.")
            return

        if not AutoTokenizer or not AutoModelForSequenceClassification:
            logger.warning("transformers or required classes unavailable — using lexicon fallback only.")
            return

        # استخدام متغير `device` المعرّف على مستوى الوحدة مباشرة
        model_device = device 
        logger.info("Loading model %s on %s...", model_name, model_device)
        # تحميل tokenizer والنموذج (قد يستغرق وقتاً، وقد يفشل إذا لم تتوفر الموارد)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
            if TORCH_AVAILABLE and torch is not None and hasattr(self.model, 'to'):
                self.model = self.model.to(model_device)
        except Exception: # Catching a broad exception is acceptable here as many things can go wrong.
            logger.exception("Failed to load ML model; continuing with fallback lexicon approach.")
            self.tokenizer = None
            self.model = None
            return
    def analyze_sentiment(self, user_input, conversation_context):
        """
        تحليل المشاعر المعقدة باستخدام السياق الطويل (ML).
        ترجع: (label, probabilities_array_or_None, risk_level)
        """
        # إذا لم يتم تحميل النموذج (أو مكتبات التعلم الآلي غير متاحة)، نرجع إلى الـ fallback
        if not getattr(self, 'model', None):
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        # 1. تحديث سياق المحادثة
        context_list = [self.preprocess_arabic_text(c) for c in (conversation_context or [])]
        context_list.append(self.preprocess_arabic_text(user_input))

        # 2. توليد ملخص السياق (لآخر N أدوار)
        relevant_context = context_list[-self.max_context_turns:]
        contextual_input = " [SEP] ".join(relevant_context)

        # 3. الترميز والتحليل
        encoding = self.tokenizer.encode_plus(
            contextual_input,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # إذا كان torch غير متوفر، نعيد fallback
        if not TORCH_AVAILABLE or torch is None:
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # 4. التنبؤ
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = getattr(outputs, 'logits', None)
        if logits is None:
            return self.fallback_sentiment(user_input), None, self.check_for_risk(user_input)

        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probabilities))
        predicted_sentiment = self.labels.get(predicted_class_id, "محايد/أخرى")

        risk_level = self.check_for_risk(user_input)

        return predicted_sentiment, probabilities, risk_level

    def fallback_sentiment(self, message):
        """
        Lexicon fallback sentiment detection (منطقه الأصلي).
        """
        text = message or ""
        norm = normalize_text(text)
        if not norm:
            return "محايد/أخرى"

        words_original = norm.split()
        words_stemmed = [simple_stem(w) for w in words_original]
        norm_polarity = {simple_stem(normalize_text(k)): v for k, v in EMOTION_POLARITY.items()}

        score = 0.0
        matches = 0
        for i, w_stem in enumerate(words_stemmed):
            polarity = 0
            if w_stem in norm_polarity:
                polarity = norm_polarity[w_stem]
            # ... (منطق difflib fuzzy matching كما هو) ...

            if polarity != 0:
                matches += 1
                if i > 0 and words_original[i - 1] in ("لا", "ليس", "لم", "لن", "غير", "ما"):
                    score -= polarity * 1.5
                else:
                    score += polarity
        
        if matches == 0:
            return "محايد/أخرى"
        
        avg_polarity = score / max(1, matches)
        
        # إصلاح منطق الـ Fallback: البحث عن أقرب شعور بناءً على درجة القطبية المحسوبة
        # بدلاً من التحويل الخاطئ لكل المشاعر السلبية إلى "حزن"
        if not EMOTION_POLARITY:
            return "محايد/أخرى"

        # البحث عن أقرب شعور في القاموس للنتيجة المحسوبة
        closest_emotion = min(
            EMOTION_POLARITY.keys(),
            key=lambda emotion: abs(EMOTION_POLARITY[emotion] - avg_polarity)
        )
        return closest_emotion

    def check_for_risk(self, text):
        """
        فحص سريع للكلمات أو العبارات التي تدل على حالة أزمة.
        """
        for word in CRISIS_KEYWORDS:
            if word in text.lower():
                return "خطورة عالية - يرجى طلب المساعدة"
        return "خطورة منخفضة"

    def preprocess_arabic_text(self, text):
        return text.replace('\n', ' ').strip()
        
    def _handle_greeting(self, normalized_text):
        """Handles greeting messages."""
        for key, resp in GREETING_RESPONSES.items():
            if key in normalized_text:
                return resp
        return None

    def _handle_phrase_bank(self, normalized_text):
        """Matches input against the phrase bank."""
        if not (isinstance(PHRASE_BANK, dict) and PHRASE_BANK):
            return None
        try:
            keys = list(PHRASE_BANK.keys())
            # Use a slightly lower cutoff to be more lenient
            match = difflib.get_close_matches(normalized_text, keys, n=1, cutoff=0.45)
            if match:
                return PHRASE_BANK.get(match[0])
        except Exception:
            logger.warning("Error during difflib matching in phrase bank.")
        return None

    def _handle_question(self, user_text):
        """Handles direct questions from the user."""
        if re.search(r'\?|\bكيف\b|\bماذا\b|\bلماذا\b|\bأين\b|\bمتى\b|\bهل\b', user_text):
            snippet = user_text if len(user_text) < 120 else user_text[:117] + '...'
            return (
                f"قرأت سؤالك: «{snippet}». أقدّر وضوحك في التعبير. يمكنني أن أقدم لك المساعدة بأحد الأشكال التالية:\n"
                "- (أ) خطوات عملية قصيرة.\n"
                "- (ب) تفسير مفصل للموضوع.\n"
                "- (ج) أمثلة وتجارب مشابهة.\n\n"
                "أي نمط تفضل أن أبدأ به الآن؟"
            )
        return None

    def _warm_signature(self):
        return "\n\nأبقى معك هنا، هل تود المتابعة الآن أم تفضّل تمرينًا قصيرًا؟"

    # دالة توليد الرد - تم تعديلها لتكون جزءاً من الفئة
    def generate_response(self, sentiment, risk_level, user_input=None, conversation_context=None):
        """
        توليد استجابة عربية فصيحة، مرتبة، ومتصلة بسياق المستخدم.
        مميزات الرد:
        - يعالج حالات الخطر فوراً.
        - يعكس ما قاله المستخدم بطريقة محترمة ومتماسكة.
        - يقدم طروحات عملية صغيرة مناسبة للشباب (تهدئة، تواصل، تمرينات قصيرة).
        - يختم بسؤال متابعة وتشجيع وعبارات ودية.
        """
        # 1) Handle crisis immediately
        if "خطورة عالية" in risk_level:
            return CRISIS_RESPONSE_AR

        user_text = (user_input or "").strip()
        norm = normalize_text(user_text)

        # 2) الاستجابات المبنية على القواعد (Greetings, Phrase Bank, Questions)
        if norm:
            greeting_resp = self._handle_greeting(norm)
            if greeting_resp:
                return f"{greeting_resp} {self._warm_signature()}"

            phrase_bank_resp = self._handle_phrase_bank(norm)
            if phrase_bank_resp:
                return f"{phrase_bank_resp} {self._warm_signature()}"

        if user_text:
            question_resp = self._handle_question(user_text)
            if question_resp:
                return f"{question_resp} {self._warm_signature()}"

        # 5) استجابات عاطفية مُفصَّلة مع مقترحات عملية صغيرة قابلة للتطبيق فوراً
        if isinstance(sentiment, str) and sentiment:
            # انعكاس موجز لكلمات المستخدم ليشعر بالملاحظة
            reflected = f"أنت قلت: «{user_text}»." if user_text else ""

            # استخدام المشاعر المحددة من النموذج مباشرة لتقديم استجابات أكثر دقة
            if "حزن" in sentiment or "وحدة" in sentiment:
                body = (
                    f"{reflected} أسمع ثقل هذه المشاعر في كلامك، وهذا أمر مفهوم تمامًا. الشعور بالوحدة مؤلم، لكن تذكر أنك لست وحدك في هذا الشعور. "
                    "لقد صممت لك رحلة صغيرة اسمها 'خطوات نحو التواصل' لمساعدتك. "
                    'هل تود أن تبدأ بأول مهمة الآن؟ إنها خطوة بسيطة جدًا للبدء.<br>'
                    '<button class="chat-btn" data-input="نعم، أريد بدء أول مهمة في رحلة التواصل">🚀 نعم، لنبدأ</button>'
                    '<button class="chat-btn" data-input="ليس الآن، ربما لاحقاً">⏱️ ليس الآن</button>'
                )
                return body + self._warm_signature()

            if "قلق" in sentiment or "توتر" in sentiment:
                body = (
                    f"{reflected} أرى توترًا أو قلقًا واضحين — وهذا شعور شائع ومؤلم. "
                    "لتخفيف الضيق فورًا، أي هذه الاقتراحات تود تجربته الآن؟<br>"
                    '<button class="chat-btn" data-input="أريد تجربة تمرين التنفس العميق">🧘 تمرين تنفس</button>'
                    '<button class="chat-btn" data-input="أريد تدوين ما يقلقني في مساحة الأفكار">📝 تدوين الأفكار</button>'
                    '<button class="chat-btn" data-input="أريد نصيحة سريعة للتهدئة">💡 نصيحة سريعة</button>'
                )
                return body + self._warm_signature()

            if "تعلق" in sentiment or "قلة ثقة" in sentiment:
                body = (
                    f"{reflected} أسمع في كلماتك أنك قد تكون في صراع مع الثقة بالنفس، وهذا أمر نواجهه جميعًا. "
                    "لدينا رحلة مصممة خصيصًا لمساعدتك على رؤية قوتك الداخلية اسمها 'تعزيز الثقة بالنفس'. "
                    'هل تود أن نبدأ بأول خطوة صغيرة معًا؟<br>'
                    '<button class="chat-btn" data-input="نعم، أريد بدء رحلة الثقة بالنفس">💪 نعم، لنبدأ</button>'
                )
                return body + self._warm_signature()

        # 6) رد شامل عام عندما لا يوجد تطابق قوي: تعاطف، عرض خيارات عملية، وسؤال متابعة
        if user_text:
            snippet = user_text if len(user_text) < 300 else user_text[:297] + '...'
            body = (
                f"شكرًا لثقتك بمشاركتك: «{snippet}». ما شاركتَه يُعطي صورة مهمة عن ما تمرّ به، وسأبقى معك خطوة بخطوة. "
                "أقترح خيارين: إما أن نتابع بالحديث لتعميق الفهم، أو أن أجرب معك تمرينًا قصيرًا الآن (تنفّس/كتابة/تواصل صغير). أيهما تفضّل؟"
            )
            return body + self._warm_signature()

        # 7)Fallback نهائي بسيط ودافئ
        return (
            "أهلاً—إن رغبت، ابدأ بمشاركة شيء واحد بسيط عن يومك أو عن شعورك الآن، حتى إن كان مجرد كلمة. "
            "أنا هنا للاستماع والدعم دون أحكام."
        )

# ----------------- تهيئة النموذج عند البدء -----------------

# تهيئة المحلل مرة واحدة فقط
# (يجب أن يتم هذا بشكل مثالي باستخدام app.before_first_request أو في سياق تشغيل Flask المناسب)
try:
    analyzer = AdvancedSentimentAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize AdvancedSentimentAnalyzer: {e}")
    analyzer = AdvancedSentimentAnalyzer(model_name=None) # تهيئة بدون نموذج للمتابعة بالـ Fallback

# ----------------- المسارات (Routes) -----------------

@app.route('/')
def index():
    # في تطبيق حقيقي، ستعرض هذه الدالة ملف HTML الواجهة الأمامية
    return render_template('index.html')


# Serve original filenames so the HTML can remain unchanged (href="style.css", src="script.js")
@app.route('/style.css')
def serve_root_style():
    return app.send_static_file('style.css')


@app.route('/script.js')
def serve_root_script():
    return app.send_static_file('script.js')

@app.route('/chat', methods=['POST'])
def chat():
    """
    مسار أساسي لتلقي مدخلات المستخدم ومعالجتها.
    """
    try:
        data = request.get_json(force=True)
        user_input = data.get('message', '')
        # سياق المحادثة يجب أن يأتي من الواجهة الأمامية (Chat History)
        context = data.get('context', []) 
        
        if not user_input:
            return jsonify({'response': "يرجى إدخال رسالة.", 'sentiment_label': 'محايد'})

        # 1. تحليل المشاعر والمخاطر
        sentiment, probabilities, risk_level = analyzer.analyze_sentiment(user_input, context)

        # 2. توليد الاستجابة (نمرّر نص المستخدم والسياق للحصول على رد ملائم)
        response_text = analyzer.generate_response(sentiment, risk_level, user_input=user_input, conversation_context=context)
        
        # 3. حفظ التفاعل
        save_interaction(user_input, response_text, sentiment)
        
        return jsonify({
            'response': response_text,
            'sentiment_label': sentiment,
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'risk_level': risk_level,
            'disclaimer': APP_DISCLAIMER
        })
        
    except Exception as e:
        logger.exception('Failed during chat processing')
        return jsonify({'response': "حدث خطأ غير متوقع. يرجى المحاولة لاحقاً.", 'sentiment_label': 'محايد', 'error': str(e)}), 500

@app.route('/save_interaction', methods=['POST'])
def save_interaction_route():
     # إعادة توجيه للوظيفة الأساسية (ليتم استخدامها عبر AJAX)
     try:
         # Using force=False is safer, but requires correct Content-Type header
         payload = request.get_json()
         if not payload:
             return {'status': 'error', 'error': 'Invalid JSON payload'}, 400
         save_interaction(
             payload.get('input', ''),
             payload.get('result', ''),
             payload.get('sentiment_label')
         )
         return {'status': 'ok'}
     except Exception as e:
         return {'status': 'error', 'error': str(e)}, 500

def save_interaction(user_input, response_text, sentiment_label):
    """
    حفظ التفاعل في ملف السجل.
    """
    entry = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'input': user_input,
        'result': response_text,
        'sentiment_label': sentiment_label
    }
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, 'interaction_log.json')
    # منطق القراءة/الكتابة كما هو في الكود الأصلي
    try:
         if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                 arr = json.load(fh) if os.path.getsize(path) > 0 else []
         else:
             arr = []
    except (json.JSONDecodeError, Exception):
         arr = []
    
    arr.append(entry)
    with open(path, 'w', encoding='utf-8') as fh:
         json.dump(arr, fh, ensure_ascii=False, indent=2)


# ----------------- تمارين يومية وتتبع التقدم -----------------
EXERCISES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'exercises.json')
PROGRESS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'progress.json')
USERS_PATH = os.path.join(os.path.dirname(__file__), 'data', 'users.json')

def load_json_file(path, default_value):
    """دالة مساعدة لقراءة ملف JSON بأمان."""
    if not os.path.exists(path):
        return default_value
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, Exception):
        logger.exception('Failed to load or parse JSON file: %s', path)
        return default_value

def load_exercises_file():
    return load_json_file(EXERCISES_PATH, [])

def load_users():
    return load_json_file(USERS_PATH, {})

def save_users(users):
    try:
        os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
        with open(USERS_PATH, 'w', encoding='utf-8') as fh:
            json.dump(users, fh, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception('Failed to save users')

def award_badges_for_user(user):
    """Simple badge awarding based on streak length."""
    badges = set(user.get('badges', []))
    streak = int(user.get('streak', 0))
    if streak >= 3:
        badges.add('3-day-streak')
    if streak >= 7:
        badges.add('7-day-streak')
    if streak >= 30:
        badges.add('30-day-streak')
    user['badges'] = sorted(list(badges))
    return user

@app.route('/exercises', methods=['GET'])
def exercises_route():
    ex = load_exercises_file()
    return jsonify({'exercises': ex})


@app.route('/user/create', methods=['POST'])
def create_user():
    try:
        payload = request.get_json(force=True)
        username = (payload.get('username') or '').strip()
        if not username:
            return jsonify({'status': 'error', 'error': 'username required'}), 400
        users = load_users()
        # generate simple id
        user_id = str(uuid.uuid4())
        users[user_id] = {'username': username, 'created': datetime.datetime.utcnow().isoformat() + 'Z'}
        # initialize streak/badges
        users[user_id].update({'streak': 0, 'last_checkin': None, 'badges': []})
        save_users(users)
        return jsonify({'status': 'ok', 'user_id': user_id, 'username': username})
    except Exception as e:
        logger.exception('Failed to create user')
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/progress', methods=['GET'])
def get_progress():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id required'}), 400
    try:
        arr = load_json_file(PROGRESS_PATH, [])
        user_records = [r for r in arr if r.get('user_id') == user_id]
        return jsonify({'status': 'ok', 'records': user_records})
    except Exception as e:
        logger.exception('Failed to load exercises')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/track_progress', methods=['POST'])
def track_progress():
    try:
        payload = request.get_json(force=True)
        record = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'user_id': payload.get('user_id'),
            'exercise_id': payload.get('exercise_id'),
            'completed': bool(payload.get('completed', False)),
            'note': payload.get('note', '')
        }
        if not record.get('user_id'):
            return jsonify({'status': 'error', 'error': 'user_id required'}), 400
        os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
        arr = load_json_file(PROGRESS_PATH, [])
        arr.append(record)
        with open(PROGRESS_PATH, 'w', encoding='utf-8') as fh:
            json.dump(arr, fh, ensure_ascii=False, indent=2)
        return jsonify({'status': 'ok', 'record': record})
    except Exception as e:
        logger.exception('Failed to track progress')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    # ... (دالة analyze_audio كما هي تقريباً - تم حذفها للاختصار) ...
    return jsonify({'status': 'error', 'error': 'Audio analysis logic not fully included in this snippet'}), 500


@app.route('/daily_quote', methods=['GET'])
def daily_quote():
    # Return a random motivational quote (can be extended to rotate per-user)
    q = random.choice(MOTIVATIONAL_QUOTES) if MOTIVATIONAL_QUOTES else "أنت تستحق لحظة لطف مع نفسك اليوم."
    return jsonify({'quote': q})

@app.route('/resources', methods=['GET'])
def get_resources():
    # Return youth resources loaded from the JSON file
    return jsonify(YOUTH_RESOURCES)

@app.route('/journey', methods=['GET'])
def get_journey_data():
    # مسار جديد لتحميل بيانات الرحلة
    journey_data = load_json_file(os.path.join(os.path.dirname(__file__), 'data', 'journeys.json'), {})
    return jsonify(journey_data)

@app.route('/user/data', methods=['GET'])
def get_user_data():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'status': 'error', 'error': 'user_id required'}), 400
    
    users = load_users()
    user = users.get(user_id)
    if not user:
        return jsonify({'status': 'error', 'error': 'user not found'}), 404
        
    # تحميل تعريفات الإنجازات لإرسالها مع بيانات المستخدم
    achievements_def = load_json_file(os.path.join(os.path.dirname(__file__), 'data', 'achievements.json'), {})
    user['achievements_details'] = [achievements_def[ach_id] for ach_id in user.get('achievements', []) if ach_id in achievements_def]

    return jsonify({'status': 'ok', 'user': user})

@app.route('/quest/complete', methods=['POST'])
def complete_quest_route():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        quest_id = payload.get('quest_id')

        if not user_id or not quest_id:
            return jsonify({'status': 'error', 'error': 'user_id and quest_id are required'}), 400

        users = load_users()
        user = users.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'error': 'user not found'}), 404

        # إضافة المهمة إلى قائمة المهام المكتملة
        progress = user.get('progress', {})
        if quest_id in progress:
            return jsonify({'status': 'ok', 'message': 'quest already completed', 'user': user})
        
        progress[quest_id] = True
        user['progress'] = progress

        # إضافة نقاط الخبرة
        journeys = load_json_file(os.path.join(os.path.dirname(__file__), 'data', 'journeys.json'), {})
        quest_xp = 0
        completed_journey_id = None
        for journey_type, journey_data in journeys.items():
            for quest in journey_data.get('quests', []):
                if quest['id'] == quest_id:
                    quest_xp = quest.get('xp', 0)
                    # التحقق مما إذا كانت الرحلة قد اكتملت
                    all_quests_in_journey = {q['id'] for q in journey_data.get('quests', [])}
                    if all_quests_in_journey.issubset(set(user['progress'].keys())):
                        completed_journey_id = journey_data.get('id')
                    break
        
        user['xp'] = user.get('xp', 0) + quest_xp

        # منح الإنجاز إذا اكتملت الرحلة
        new_achievement = None
        if completed_journey_id:
            user_achievements = set(user.get('achievements', []))
            if completed_journey_id not in user_achievements:
                user_achievements.add(completed_journey_id)
                user['achievements'] = sorted(list(user_achievements))
                achievements_def = load_json_file(os.path.join(os.path.dirname(__file__), 'data', 'achievements.json'), {})
                new_achievement = achievements_def.get(completed_journey_id)

        save_users(users)
        return jsonify({'status': 'ok', 'user': user, 'new_achievement': new_achievement})

    except Exception as e:
        logger.exception('Failed to complete quest')
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/daily_checkin', methods=['POST'])
def daily_checkin():
    try:
        payload = request.get_json(force=True)
        user_id = payload.get('user_id')
        if not user_id:
            return jsonify({'status': 'error', 'error': 'user_id required'}), 400
        users = load_users()
        user = users.get(user_id)
        if not user:
            return jsonify({'status': 'error', 'error': 'user not found'}), 404

        today = datetime.datetime.utcnow().date()
        last = None
        if user.get('last_checkin'):
            try:
                last = datetime.datetime.fromisoformat(user.get('last_checkin')).date()
            except Exception:
                last = None

        if last == today:
            # already checked in today
            return jsonify({'status': 'ok', 'message': 'already_checked_in', 'streak': user.get('streak', 0), 'badges': user.get('badges', [])})

        # increment streak if yesterday was last checkin, otherwise reset to 1
        if last == (today - datetime.timedelta(days=1)):
            user['streak'] = int(user.get('streak', 0)) + 1
        else:
            user['streak'] = 1

        user['last_checkin'] = datetime.datetime.utcnow().isoformat()
        user = award_badges_for_user(user)
        users[user_id] = user
        save_users(users)

        return jsonify({'status': 'ok', 'message': 'checked_in', 'streak': user['streak'], 'badges': user.get('badges', [])})
    except Exception as e:
        logger.exception('daily_checkin failed')
        return jsonify({'status': 'error', 'error': str(e)}), 500


# --- 4. تشغيل التطبيق (Execution) ---

if __name__ == "__main__":
    # تحميل البيانات الخارجية عند بدء التشغيل
    default_data = {
        "greetings": GREETING_RESPONSES,
        "emotions": EMOTION_RESPONSES,
        "special": SPECIAL_RESPONSES,
        "examples": EXAMPLE_PHRASES,
        "phrase_bank": PHRASE_BANK,
        "quotes": MOTIVATIONAL_QUOTES,
        "resources": YOUTH_RESOURCES
    }
    GREETING_RESPONSES, EMOTION_RESPONSES, SPECIAL_RESPONSES, EXAMPLE_PHRASES, PHRASE_BANK, MOTIVATIONAL_QUOTES, YOUTH_RESOURCES = load_external_responses(default_data)
    # يفضل استخدام gunicorn أو waitress في الإنتاج
    host = '0.0.0.0'
    port = 5000
    print(f"Starting Flask app on http://{host}:{port}/ — bind to all interfaces for local testing")
    app.run(host=host, port=port, debug=True)