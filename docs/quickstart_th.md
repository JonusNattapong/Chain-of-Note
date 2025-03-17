# คู่มือเริ่มต้นใช้งาน Chain-of-Note RAG (ฉบับภาษาไทย)

คู่มือนี้ให้คำแนะนำทีละขั้นตอนเพื่อเริ่มต้นใช้งานระบบ Chain-of-Note RAG สำหรับลดอาการหลอน (Hallucinations) ในคำตอบที่สร้างโดย AI

## การติดตั้ง

1.  โคลนรีพอสิทอรี:

    ```bash
    git clone https://github.com/your-username/chain-of-note.git
    cd chain-of-note
    ```

    แทนที่ `https://github.com/your-username/chain-of-note.git` ด้วย URL ของรีพอสิทอรีจริง

2.  ติดตั้งแพ็คเกจที่จำเป็น:

    ```bash
    pip install -r requirements.txt
    ```

    หรือคุณสามารถติดตั้งแพ็คเกจด้วยตนเอง:

    ```bash
     pip install transformers>=4.30.0 datasets>=2.12.0 sentence-transformers>=2.2.2 faiss-cpu>=1.7.4 torch>=2.0.0 numpy>=1.24.0 langchain>=0.0.267 pandas>=2.0.0 tqdm>=4.65.0 python-dotenv>=1.0.0
    ```

## การใช้งานเบื้องต้น

นี่คือตัวอย่างง่ายๆ เพื่อสาธิตการใช้งานพื้นฐานของระบบ Chain-of-Note RAG:

```python
from src.rag_system import ChainOfNoteRAG
from examples.sample_data import get_sample_documents, get_sample_queries

# เริ่มต้นระบบ RAG
rag_system = ChainOfNoteRAG()

# โหลดเอกสารตัวอย่าง
sample_documents = get_sample_documents()
rag_system.add_documents(sample_documents)

# รับคำถามตัวอย่าง
sample_queries = get_sample_queries()

# ประมวลผลคำถามตัวอย่าง
query = sample_queries[0]["query"]
response = rag_system.query(query, top_k=3, return_context=True, return_notes=True)

# พิมพ์ผลลัพธ์
print(f"คำถาม: {query}")
print(f"คำตอบ: {response['answer']}")
print("บันทึก:")
print(response["notes"])
print("บริบท:")
for doc in response["context"]:
    print(f"- {doc['content'][:100]}...") # พิมพ์ 100 ตัวอักษรแรกของแต่ละเอกสาร
```

## ส่วนประกอบหลัก

ระบบ Chain-of-Note RAG ประกอบด้วยส่วนประกอบหลักดังต่อไปนี้:

*   **`ChainOfNoteRAG`**: คลาสหลักที่จัดการกระบวนการ RAG ทั้งหมด  เริ่มต้นโมเดล Embedding, Document Store และส่วนประกอบ Chain-of-Note
    *   `add_documents(documents)`: เพิ่มเอกสารในระบบ
    *   `query(query, top_k=5, return_context=False, return_notes=False)`: ประมวลผลคำถามและส่งคืนคำตอบที่สร้างขึ้น พร้อมตัวเลือกในการรวมบริบทที่ดึงมาและบันทึกที่สร้างขึ้น
*   **`EmbeddingModel`**: อินเทอร์เฟซสำหรับโมเดล Embedding การใช้งานเริ่มต้นคือ `SentenceTransformerEmbeddings`
*   **`DocumentStore`**: จัดเก็บ Embedding ของเอกสารและมีวิธีการสำหรับค้นหาเอกสารที่คล้ายกัน
*   **`ChainOfNote`**: สร้างคำตอบสุดท้ายตามเอกสารที่ดึงมาและคำถามของผู้ใช้ โดยใช้เทคนิค Chain-of-Note

## การเรียกใช้ Demo

คุณสามารถเรียกใช้การสาธิตระบบทั้งหมดได้โดยใช้สคริปต์ `demo.py` ที่ให้มา:

```bash
python examples/demo.py
```

สคริปต์นี้จะ:

1.  เริ่มต้นระบบ RAG
2.  โหลดเอกสารตัวอย่าง
3.  ประมวลผลคำถามตัวอย่าง
4.  แสดงคำตอบที่สร้างขึ้น เอกสารที่ดึงมา และบันทึกขั้นกลาง

## การใช้งานขั้นสูง

ไฟล์ `src/advanced_techniques.py` มีตัวอย่างรูปแบบการใช้งานขั้นสูงเพิ่มเติม ไดเรกทอรี `examples/` มีตัวอย่างเพิ่มเติม รวมถึงบทแนะนำ Jupyter Notebook (`jupyter_tutorial.ipynb`) และตัวอย่างการใช้งานจริง (`real_world_example.py`)

## การอ่านเพิ่มเติม

*   **ตัวอย่าง:** ไดเรกทอรี `examples/` มีตัวอย่างเพิ่มเติม รวมถึงบทแนะนำ Jupyter Notebook (`jupyter_tutorial.ipynb`), ตัวอย่างการใช้งานจริง (`real_world_example.py`) และสคริปต์สาธิต (`demo.py`)
*   **เอกสารอ้างอิง API:** สำหรับข้อมูล API โดยละเอียด ให้สำรวจซอร์สโค้ดโดยตรง โดยเฉพาะไดเรกทอรี `src/`
*   **การเปรียบเทียบ:** ไฟล์ `examples/comparison.py` และ `src/evaluation.py` มีการเปรียบเทียบกับ RAG มาตรฐาน

## ดูเพิ่มเติม
* [คู่มือระบบ](system_guide_th.md)
* [เอกสารอ้างอิง API](api_reference_th.md)

## ดูเพิ่มเติม
* [คู่มือระบบ](system_guide_th.md)
* [เอกสารอ้างอิง API](api_reference_th.md)