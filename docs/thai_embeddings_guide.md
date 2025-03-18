# คู่มือการใช้งานระบบ Embeddings

## ภาพรวมของระบบ
ระบบ Chain-of-Note รองรับการใช้งาน Embeddings จากสองแหล่งหลัก:
1. HuggingFace
2. Mistral AI

## การทำงานของระบบ Embeddings

### 1. โครงสร้างของระบบ
```
src/
  ├── embeddings.py     # คลาสหลักสำหรับ embeddings
  ├── rag_system.py     # ระบบ RAG หลัก
  └── chain_of_note.py  # ตัวประมวลผล Chain-of-Note
```

### 2. คลาส Embeddings หลัก

#### 2.1 SentenceTransformerEmbeddings
```python
class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN not found")
        self.model = SentenceTransformer(model_name, token=token)
```
- ใช้สำหรับสร้าง embeddings ด้วย HuggingFace
- รองรับหลายโมเดล เช่น all-mpnet-base-v2, all-MiniLM-L6-v2
- ต้องการ HuggingFace Token

#### 2.2 MistralEmbeddings
```python
class MistralEmbeddings(EmbeddingModel):
    def __init__(self):
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found")
        self.client = MistralClient(api_key=MISTRAL_API_KEY)
```
- ใช้ API ของ Mistral AI
- ใช้โมเดล mistral-embed เป็นค่าเริ่มต้น
- ต้องการ Mistral API Key

### 3. การตั้งค่าผ่านเว็บแอปพลิเคชัน

#### 3.1 การกำหนดค่า API
- เลือกผู้ให้บริการ (HuggingFace/Mistral AI)
- กรอก API Token
- ระบบจะบันทึกใน environment variables

```python
api_provider = st.sidebar.selectbox(
    "Embedding Provider",
    ["HuggingFace", "Mistral AI"]
)

if api_provider == "HuggingFace":
    hf_token = st.sidebar.text_input("HuggingFace Token", type="password")
    if hf_token:
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
else:
    mistral_token = st.sidebar.text_input("Mistral API Key", type="password")
    if mistral_token:
        os.environ["MISTRAL_API_KEY"] = mistral_token
```

#### 3.2 การเลือกโมเดล
- HuggingFace: เลือกได้หลายโมเดล
- Mistral AI: ใช้ mistral-embed เป็นค่าเริ่มต้น

### 4. การทำงานของ Embeddings

#### 4.1 การสร้าง Embeddings
```python
def embed_documents(self, documents: List[str]) -> np.ndarray:
    """
    แปลงเอกสารเป็น vectors
    
    Args:
        documents: รายการข้อความ
        
    Returns:
        numpy array ของ embeddings
    """
```

#### 4.2 การค้นหาเอกสารที่เกี่ยวข้อง
```python
def query(self, query: str, top_k: int = 3) -> List[Dict]:
    """
    ค้นหาเอกสารที่เกี่ยวข้องกับคำถาม
    
    Args:
        query: คำถาม
        top_k: จำนวนเอกสารที่ต้องการ
        
    Returns:
        รายการเอกสารที่เกี่ยวข้อง
    """
```

## การใช้งานระบบ

### 1. การติดตั้ง
```bash
pip install -r requirements.txt
```

### 2. การตั้งค่า API Keys
สร้างไฟล์ .env:
```
HUGGINGFACE_TOKEN=your_token_here
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. การรันระบบ
```bash
python web.py
```

## การพัฒนาต่อยอด

### 1. การเพิ่ม Embedding Provider ใหม่
1. สร้างคลาสใหม่ที่สืบทอดจาก EmbeddingModel
2. implement วิธี embed_documents และ embed_query
3. เพิ่มตัวเลือกใน web.py

### 2. การปรับแต่งพารามิเตอร์
- ปรับ chunk_size สำหรับการแบ่งเอกสาร
- ปรับ top_k สำหรับการค้นหา
- เพิ่มการกรองผลลัพธ์

### 3. การวัดผล
- ความแม่นยำของการค้นหา
- ความเร็วในการประมวลผล
- การใช้ทรัพยากร

## ข้อควรระวัง

1. ความปลอดภัย
- เก็บ API Keys อย่างปลอดภัย
- ใช้ HTTPS สำหรับการสื่อสาร
- ระวังการรั่วไหลของข้อมูล

2. การใช้งาน API
- ตรวจสอบโควต้าการใช้งาน
- จัดการ rate limiting
- มีระบบ fallback

3. ประสิทธิภาพ
- เลือกขนาดโมเดลให้เหมาะสม
- ใช้ caching เมื่อจำเป็น
- ติดตามการใช้ทรัพยากร

## แนวทางการวิจัยต่อยอด

1. การเพิ่มประสิทธิภาพ
- ทดลองใช้เทคนิค quantization
- พัฒนาวิธีการ caching
- ปรับปรุงการค้นหาด้วย approximate nearest neighbors

2. การปรับปรุงคุณภาพ
- พัฒนาวิธีการ reranking
- เพิ่มความหลากหลายของผลลัพธ์
- ปรับปรุงการจัดการกับภาษาไทย

3. การประยุกต์ใช้
- ระบบแนะนำเอกสาร
- การวิเคราะห์ความคิดเห็น
- การสรุปความอัตโนมัติ