to run first write ollama serve then 
uvicorn app:app --host 127.0.0.1 --port 8000

If you WANT step-by-step explanation ability

You need to slightly relax the prompt.

🔧 Change this:
- Do NOT make up solutions that aren't supported by the records.
✅ Replace with:
- Prefer solutions from records.
- If needed, explain steps using general engineering knowledge.
- Do not hallucinate unsafe or unrealistic fixes.


Backend Changes (app.py)

Add a new endpoint:

class TranslateRequest(BaseModel):
    text: str
    target_lang: str


@app.post("/translate")
async def translate(req: TranslateRequest):
    prompt = f"""
Translate the following text into {req.target_lang}.
Only return translated text.

Text:
{req.text}
"""


    translated = await ask_ollama(prompt)
    return {"translated_text": translated}




Frontend Changes (index.html)

In your AI result section, add buttons.

🔧 Modify this part in :

Find:

<div class="ai-body">${esc(ai_suggestion)}</div>
✅ Replace with:
<div class="ai-body" id="ai-text">${esc(ai_suggestion)}</div>

<div style="padding: 12px 18px; display:flex; gap:10px;">
  <button class="btn" onclick="translateText('English')">English</button>
  <button class="btn" onclick="translateText('Hindi')">Hindi</button>
  <button class="btn" onclick="translateText('Korean')">Korean</button>
</div>
⚙️ Add JS function

Add this in your <script>:

let originalText = '';

function renderResults(query, data) {
  const { ai_suggestion, matched_records } = data;
  originalText = ai_suggestion;  // store original

  // rest stays same...
}
Add translation function:
async function translateText(lang) {
  const textBox = document.getElementById('ai-text');

  if (lang === "English") {
    textBox.innerText = originalText;
    return;
  }

  textBox.innerText = "Translating...";

  try {
    const r = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: originalText,
        target_lang: lang
      })
    });

    const data = await r.json();
    textBox.innerText = data.translated_text;

  } catch {
    textBox.innerText = "Translation failed.";
  }
}



important optional
adding a prompt
If the user asks about a step in a solution, explain it clearly.






✅ What you will do (end-to-end)
📦 1. Transfer your zip

Move your project zip to your work laptop and extract it:

project/
🧱 2. Open terminal inside that folder
cd path\to\project
🔧 3. Create virtual environment
python -m venv venv
▶️ 4. Activate it
venv\Scripts\activate
🔍 5. Confirm version
python --version

👉 Must be:

Python 3.11.x ✅
📥 6. Install dependencies (OFFLINE)
pip install --no-index --find-links=offline_packages -r requirements-lock.txt



def load_and_clean(filepath: str) -> pd.DataFrame:
    print(f"[1/3] Reading CSV: {filepath}")

    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip()

    required = [COL_SMD_LINE, COL_MACHINE, COL_PROBLEM, COL_SOLUTION]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing column: '{col}'\nAvailable: {list(df.columns)}"
            )
