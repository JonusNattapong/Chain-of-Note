# ระบบ Chain-of-Note RAG

โปรเจกต์นี้เป็นการนำเสนอระบบ Retrieval-Augmented Generation (RAG) ที่ใช้เทคนิค Chain-of-Note เพื่อลดอาการ Hallucination ในการตอบสนองที่สร้างโดย AI

## ภาพรวม

Chain-of-Note RAG เป็นวิธีการขั้นสูงของ Retrieval-Augmented Generation ที่ช่วยลดอาการ Hallucination ได้อย่างมาก โดยการเพิ่มขั้นตอนการจดบันทึกระหว่างการดึงข้อมูลเอกสารและการสร้างคำตอบ ระบบจะสร้างบันทึกรายละเอียดจากเอกสารที่ดึงมาก่อน จากนั้นจึงใช้บันทึกเหล่านั้นเพื่อสร้างคำตอบที่ถูกต้อง

### ประโยชน์ของ Chain-of-Note เหนือกว่า RAG แบบดั้งเดิม:

1.  **ลดอาการ Hallucination**: โดยการบันทึกข้อเท็จจริงหลักจากเอกสารที่ดึงมาในบันทึก ขั้นตอนกลาง โมเดลมีโอกาสน้อยที่จะสร้างข้อมูลที่ไม่ถูกต้อง
2.  **การระบุแหล่งที่มาที่ดีขึ้น**: ขั้นตอนการจดบันทึกช่วยติดตามแหล่งที่มาของข้อมูล ทำให้คำตอบของระบบมีความโปร่งใสและตรวจสอบได้มากขึ้น
3.  **การให้เหตุผลที่เพิ่มขึ้น**: การแบ่งกระบวนการออกเป็นการดึงข้อมูล → การจดบันทึก → การสร้างคำตอบ จะสร้างห่วงโซ่การให้เหตุผลทีละขั้นตอนที่ให้ผลลัพธ์ที่แม่นยำยิ่งขึ้น

## คุณสมบัติ

-   การทำดัชนีเอกสารและการดึงข้อมูลโดยใช้ Sentence Embeddings
-   การสร้าง Chain-of-Note เพื่อการให้เหตุผลที่ดีขึ้น
-   ลดอาการ Hallucination ผ่านการเพิ่มบริบทจากบันทึก
-   การผสานรวมกับ Hugging Face Models
-   ตัวชี้วัดการประเมินที่ครอบคลุมเพื่อเปรียบเทียบประสิทธิภาพกับ RAG มาตรฐาน

## การติดตั้ง

```bash
pip install -r requirements.txt
```

สำหรับการติดตั้งเพื่อการพัฒนา:

```bash
pip install -e .
```

## การใช้งาน

### เริ่มต้นอย่างรวดเร็ว

```python
from src.data_loader import DocumentLoader
from src.embeddings import SentenceTransformerEmbeddings
from src.rag_system import ChainOfNoteRAG

# 1. เริ่มต้นระบบ RAG
embedding_model = SentenceTransformerEmbeddings()
rag_system = ChainOfNoteRAG(embedding_model=embedding_model)

# 2. โหลดเอกสาร
loader = DocumentLoader()
loader.load_text("เนื้อหาเอกสารของคุณที่นี่", {"source": "ตัวอย่าง"})
rag_system.process_documents_from_loader(loader)

# 3. สอบถามระบบ
response = rag_system.query(
    "คำถามของคุณที่นี่",
    top_k=5,
    return_context=True,
    return_notes=True
)

# 4. เข้าถึงผลลัพธ์
notes = response["notes"]
answer = response["answer"]
context = response["context"]
```

### ตัวอย่าง

โปรเจกต์นี้มีตัวอย่างหลายตัวอย่าง:

1.  **การสาธิตพื้นฐาน**: `examples/demo.py` - สาธิตฟังก์ชันหลักด้วยข้อมูลตัวอย่าง
2.  **การเปรียบเทียบ RAG**: `examples/comparison.py` - เปรียบเทียบ RAG มาตรฐานกับ Chain-of-Note RAG
3.  **ตัวอย่างการใช้งานจริง**: `examples/real_world_example.py` - ใช้ข้อมูล Wikipedia เพื่อแสดงการใช้งานจริง

## สถาปัตยกรรม

ระบบประกอบด้วยองค์ประกอบหลักหลายส่วน:

1.  **Document Loader**: จัดการการโหลดและการแบ่งเอกสารออกเป็นส่วนๆ
2.  **Embedding Model**: สร้างเวกเตอร์แทนเอกสารและคำถาม
3.  **Document Store**: ฐานข้อมูลเวกเตอร์สำหรับการดึงเอกสารอย่างมีประสิทธิภาพ
4.  **Chain-of-Note**: ใช้กระบวนการจดบันทึกและสร้างคำตอบ
5.  **RAG System**: ประสานงานกระบวนการทั้งหมดตั้งแต่คำถามจนถึงคำตอบ

### แผนภาพระบบ

![แผนภาพระบบ](public/diagram.png)

## การประเมินผล

ระบบมีโมดูลการประเมินที่วัด:

-   คะแนน Hallucination ตาม n-gram overlap
-   ความเกี่ยวข้องของคำตอบกับคำถาม
-   คะแนน ROUGE สำหรับคุณภาพของคำตอบ
-   ตัวชี้วัดเปรียบเทียบระหว่าง RAG มาตรฐานและ Chain-of-Note RAG

## การอ้างอิง

หากคุณใช้โค้ดนี้ในงานวิจัยของคุณ โปรดอ้างอิง:

```
@software{chain_of_note_rag,
  author = {zombitx64},
  title = {Chain-of-Note RAG System},
  year = {2023},
  url = {https://github.com/JonusNattapong/Chain-of-Note}
}
```

## สิทธิ์การใช้งาน

โปรเจกต์นี้ได้รับอนุญาตภายใต้ MIT License - ดูรายละเอียดได้ที่ไฟล์ LICENSE

## Changelog

See the [CHANGELOG](docs/CHANGELOG.md) for a history of changes to this project.
