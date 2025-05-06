import streamlit as st
import io
import PyPDF2
from docx import Document
from transformers import T5Tokenizer, T5ForConditionalGeneration
from googletrans import Translator
import time

languages = [
    "Afrikaans", "Akan", "Albanian", "Amharic", "Arabic", "Armenian", "Assamese", "Aymara", "Azerbaijani", "Bambara",
    "Bangla", "Basque", "Belarusian", "Bhojpuri", "Bosnian", "Bulgarian", "Burmese", "Catalan", "Cebuano",
    "Central Kurdish", "Chinese (Simplified)", "Chinese (Traditional)", "Corsican", "Croatian", "Czech", "Danish",
    "Divehi", "Dogri", "Dutch", "English", "Esperanto", "Estonian", "Ewe", "Filipino", "Finnish", "French", "Galician",
    "Ganda", "Georgian", "German", "Goan Konkani", "Greek", "Guarani", "Gujarati", "Haitian Creole", "Hausa",
    "Hawaiian", "Hebrew", "Hindi", "Hmong", "Hungarian", "Icelandic", "Igbo", "Iloko", "Indonesian", "Irish",
    "Italian", "Japanese", "Javanese", "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Krio", "Kurdish",
    "Kyrgyz", "Lao", "Latin", "Latvian", "Lingala", "Lithuanian", "Luxembourgish", "Macedonian", "Maithili",
    "Malagasy", "Malay", "Malayalam", "Maltese", "Manipuri (Meitei Mayek)", "Māori", "Marathi", "Mizo", "Mongolian",
    "Nepali", "Northern Sotho", "Norwegian", "Nyanja", "Odia", "Oromo", "Pashto", "Persian", "Polish", "Portuguese",
    "Punjabi", "Quechua", "Romanian", "Russian", "Samoan", "Sanskrit", "Scottish Gaelic", "Serbian", "Shona",
    "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "Southern Sotho", "Spanish", "Sundanese", "Swahili",
    "Swedish", "Tajik", "Tamil", "Tatar", "Telugu", "Thai", "Tigrinya", "Tsonga", "Turkish", "Turkmen", "Ukrainian",
    "Urdu", "Uyghur", "Uzbek", "Vietnamese", "Welsh", "Western Frisian", "Xhosa", "Yiddish", "Yoruba", "Zulu"
]

lang_code_map = {   "Afrikaans": "af",
    "Albanian": "sq",
    "Amharic": "am",
    "Arabic": "ar",
    "Armenian": "hy",
    "Azerbaijani": "az",
    "Basque": "eu",
    "Belarusian": "be",
    "Bengali (Bangla)": "bn",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Burmese": "my",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "Corsican": "co",
    "Croatian": "hr",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Estonian": "et",
    "Filipino": "tl",
    "Finnish": "fi",
    "French": "fr",
    "Galician": "gl",
    "Georgian": "ka",
    "German": "de",
    "Greek": "el",
    "Gujarati": "gu",
    "Haitian Creole": "ht",
    "Hausa": "ha",
    "Hawaiian": "haw",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hmong": "hmn",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Igbo": "ig",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jw",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Khmer": "km",
    "Kinyarwanda": "rw",
    "Korean": "ko",
    "Kurdish (Kurmanji)": "ku",
    "Kyrgyz": "ky",
    "Lao": "lo",
    "Latin": "la",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malay": "ms",
    "Malayalam": "ml",
    "Maltese": "mt",
    "Maori": "mi",
    "Marathi": "mr",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Norwegian": "no",
    "Odia (Oriya)": "or",
    "Pashto": "ps",
    "Persian (Farsi)": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro",
    "Russian": "ru",
    "Samoan": "sm",
    "Scottish Gaelic": "gd",
    "Serbian": "sr",
    "Shona": "sn",
    "Sindhi": "sd",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Spanish": "es",
    "Sundanese": "su",
    "Swahili": "sw",
    "Swedish": "sv",
    "Tajik": "tg",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yiddish": "yi",
    "Yoruba": "yo",
    "Zulu": "zu"    
}

st.set_page_config(page_title="LawLens: AI Tool for legal documents", layout="wide", page_icon="⚖️")

st.title("⚖️ LawLens: Legal Ease  and language Translator")

@st.cache_resource(show_spinner=" Running AI Model... ")
def load_model():
    try:
        model_path = "C:/Users/Yuvraj/Downloads/flan-t5-ilc-legal-20250430T181734Z-001/flan-t5-ilc-legal"  # Path to your trained model
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def extract_text_from_docx(file):
    doc = Document(io.BytesIO(file.read()))
    return "\n".join([p.text for p in doc.paragraphs])

def generate_summary(text, tokenizer, model, max_length=512):
    inputs = tokenizer(
        "summarize this legal document: " + text,
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=150,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_text(text, target_lang):
    try:
        translator = Translator()
        translated = translator.translate(text, dest=lang_code_map[target_lang])
        return translated.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text


tabs = st.tabs([" Legal Document Summarizer", " Translator"])

with tabs[0]:
    st.header("Legal Document Summarizer")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload legal document (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"],
            help="Upload your legal documents for summarization"
        )
    with col2:
        summary_length = st.slider("Summary Length", 150, 512, help="Adjust the length of the summary")
    
    translate_summary = st.checkbox("Translate summary", value=False)
    if translate_summary:
        target_lang = st.selectbox("Select target language", options=languages, index=0)

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        text = ""
        
        try:
            if file_type == 'pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            elif file_type == 'docx':
                text = extract_text_from_docx(uploaded_file)
            else:  # txt
                text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()

        if not text.strip():
            st.warning("No readable text found in the document")
        else:
            with st.expander("🔍 View extracted text"):
                st.text_area("Document Text", text, height=300, label_visibility="collapsed")

            if st.button("Generate Summary", type="primary"):
                tokenizer, model = load_model()
                if tokenizer and model:
                    with st.spinner("Generating summary..."):
                        start_time = time.time()
                        summary = generate_summary(text, tokenizer, model, summary_length)
                        processing_time = time.time() - start_time
                        
                        if translate_summary:
                            with st.spinner("Translating summary..."):
                                summary = translate_text(summary, target_lang)
                        
                        st.success(f"Summary generated in {processing_time:.2f} seconds")
                        st.markdown("### 📝 Summary")
                        st.write(summary)
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download Summary",
                                summary,
                                file_name=f"summary_{target_lang if translate_summary else 'en'}.txt",
                                mime="text/plain"
                            )
                        with col2:
                            st.download_button(
                                "Download Full Text",
                                text,
                                file_name="original_document.txt",
                                mime="text/plain"
                            )
                else:
                    st.error("Model not loaded properly")

# TAB 2: Translator
with tabs[1]:
    st.header("Document Translator")
    
    sub_tabs = st.tabs(["✏️ Text Translation", "📄 Document Translation"])
    
    with sub_tabs[0]:
        st.subheader("Text Translation")
        source_text = st.text_area("Enter text to translate:", height=200)
        target_lang = st.selectbox("Target language:", languages, key="text_lang")
        
        if st.button("Translate Text", key="text_trans_btn"):
            if source_text.strip():
                with st.spinner("Translating..."):
                    translated = translate_text(source_text, target_lang)
                    st.text_area("Translation:", translated, height=200)
            else:
                st.warning("Please enter some text to translate")
    
    with sub_tabs[1]:
        st.subheader("Document Translation")
        doc_file = st.file_uploader(
            "Upload document (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"],
            key="doc_trans"
        )
        doc_target_lang = st.selectbox("Target language:", languages, key="doc_lang")
        
        if st.button("Translate Document", key="doc_trans_btn"):
            if doc_file:
                file_type = doc_file.name.split('.')[-1].lower()
                text = ""
                
                try:
                    if file_type == 'pdf':
                        pdf_reader = PyPDF2.PdfReader(doc_file)
                        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    elif file_type == 'docx':
                        text = extract_text_from_docx(doc_file)
                    else:  # txt
                        text = doc_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.stop()
                
                if text.strip():
                    with st.spinner("Translating document..."):
                        translated = translate_text(text, doc_target_lang)
                        st.text_area("Translated Document", translated, height=300)
                        
                        st.download_button(
                            "Download Translation",
                            translated,
                            file_name=f"translated_{doc_target_lang}.txt",
                            mime="text/plain"
                        )
                else:
                    st.warning("No readable text found in the document")
            else:
                st.warning("Please upload a document first")


with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("""
    AI Model Settings:
    - Max summary length: Adjust in main panel
    - Optimized for legal documents
    """)
    
    st.markdown("""
    Translation Notes:
    - Powered by Google Translate
    - Some legal terms may not translate perfectly
    """)
    
    st.markdown("""
    Performance Tips:
    - For long documents, split into sections
    - PDFs with scanned text won't work
    - Complex formatting may be lost
    """)
