# ตัวอย่างการใช้งานและการประยุกต์ใช้ระบบ Chain-of-Note

## 1. การใช้งานพื้นฐาน

### 1.1 การเริ่มต้นใช้งาน
```python
# ตัวอย่างการใช้งานพื้นฐาน
from src.embeddings import SentenceTransformerEmbeddings, MistralEmbeddings
from src.rag_system import ChainOfNoteRAG

# เลือกใช้ HuggingFace
embedding_model = SentenceTransformerEmbeddings()
rag_system = ChainOfNoteRAG(embedding_model=embedding_model)

# หรือเลือกใช้ Mistral AI
embedding_model = MistralEmbeddings()
rag_system = ChainOfNoteRAG(embedding_model=embedding_model)

# เพิ่มเอกสาร
documents = [
    {
        "content": "เนื้อหาเอกสาร 1",
        "metadata": {"source": "แหล่งที่มา 1", "topic": "หัวข้อ 1"}
    },
    {
        "content": "เนื้อหาเอกสาร 2",
        "metadata": {"source": "แหล่งที่มา 2", "topic": "หัวข้อ 2"}
    }
]

rag_system.add_documents(documents)

# ค้นหาและสร้างคำตอบ
response = rag_system.query(
    "คำถามของคุณ",
    top_k=3,
    return_context=True,
    return_notes=True
)
```

### 1.2 การกำหนดค่าขั้นสูง
```python
# การกำหนดค่าการทำงาน
config = {
    "chunk_size": 500,        # ขนาดของการแบ่งเอกสาร
    "chunk_overlap": 50,      # ความซ้ำซ้อนระหว่าง chunks
    "similarity_threshold": 0.7,  # ค่าความเหมือนขั้นต่ำ
    "max_tokens": 2000        # จำนวนโทเค็นสูงสุด
}

rag_system = ChainOfNoteRAG(
    embedding_model=embedding_model,
    **config
)
```

## 2. ตัวอย่างการใช้งานจริง

### 2.1 ระบบตอบคำถามเอกสาร
```python
# ตัวอย่างระบบตอบคำถามจากเอกสารภายในองค์กร
from src.data_loader import DocumentLoader

# โหลดเอกสาร
loader = DocumentLoader()
loader.load_directory("path/to/documents")

# สร้าง embeddings และเพิ่มเข้าระบบ
rag_system.process_documents_from_loader(loader)

# ตัวอย่างการใช้งาน
questions = [
    "นโยบายการลาของบริษัทเป็นอย่างไร?",
    "ขั้นตอนการเบิกค่าใช้จ่ายมีอะไรบ้าง?",
    "การประเมินผลงานประจำปีมีเกณฑ์อะไรบ้าง?"
]

for question in questions:
    response = rag_system.query(question)
    print(f"คำถาม: {question}")
    print(f"คำตอบ: {response['answer']}\n")
```

### 2.2 ระบบวิเคราะห์ความคิดเห็น
```python
# ตัวอย่างการวิเคราะห์ความคิดเห็นลูกค้า
def analyze_customer_feedback(feedback_list):
    # เพิ่มเอกสารความคิดเห็น
    documents = [
        {"content": feedback, "metadata": {"type": "customer_feedback"}}
        for feedback in feedback_list
    ]
    rag_system.add_documents(documents)
    
    # วิเคราะห์ประเด็นสำคัญ
    analysis_questions = [
        "ประเด็นปัญหาที่ลูกค้าพบบ่อยที่สุดคืออะไร?",
        "ข้อเสนอแนะเชิงบวกที่ได้รับมีอะไรบ้าง?",
        "แนวทางการปรับปรุงที่ควรทำเร่งด่วนคืออะไร?"
    ]
    
    results = {}
    for question in analysis_questions:
        response = rag_system.query(question)
        results[question] = response['answer']
    
    return results
```

## 3. การพัฒนาต่อยอด

### 3.1 การสร้าง Custom Embedding Model
```python
from src.embeddings import EmbeddingModel
import numpy as np

class CustomThaiEmbeddings(EmbeddingModel):
    def __init__(self, model_path: str):
        # โค้ดสำหรับโหลดโมเดลภาษาไทย
        self.model = load_thai_model(model_path)
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        # ปรับแต่งการสร้าง embeddings สำหรับภาษาไทย
        embeddings = []
        for doc in documents:
            # ทำ preprocessing สำหรับภาษาไทย
            processed_doc = self.preprocess_thai_text(doc)
            # สร้าง embedding
            embedding = self.model.encode(processed_doc)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def preprocess_thai_text(self, text: str) -> str:
        # ฟังก์ชันสำหรับ preprocessing ภาษาไทย
        # (ตัดคำ, ลบคำฟุ่มเฟือย, ฯลฯ)
        pass
```

### 3.2 การเพิ่มฟีเจอร์ใหม่
```python
class EnhancedRAGSystem(ChainOfNoteRAG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    def query(self, question: str, **kwargs):
        # เพิ่มระบบ cache
        if question in self.cache:
            return self.cache[question]
        
        # เพิ่มการวิเคราะห์เพิ่มเติม
        response = super().query(question, **kwargs)
        
        # เพิ่มการประมวลผลหลังได้คำตอบ
        enhanced_response = self.post_process_response(response)
        
        # เก็บใน cache
        self.cache[question] = enhanced_response
        return enhanced_response
    
    def post_process_response(self, response):
        # เพิ่มการวิเคราะห์เพิ่มเติม
        # เช่น การสรุปความ การแปลภาษา ฯลฯ
        pass
```

## 4. การทดสอบและประเมินผล

### 4.1 การทดสอบประสิทธิภาพ
```python
from src.evaluation import RAGEvaluator

def evaluate_system(rag_system, test_questions, ground_truth):
    evaluator = RAGEvaluator()
    
    results = []
    for question, truth in zip(test_questions, ground_truth):
        response = rag_system.query(question)
        score = evaluator.evaluate_response(
            response['answer'],
            truth,
            response.get('context', [])
        )
        results.append(score)
    
    # คำนวณค่าเฉลี่ย
    avg_score = sum(results) / len(results)
    return avg_score
```

### 4.2 การทดสอบการใช้ทรัพยากร
```python
import time
import psutil
import numpy as np

def benchmark_system(rag_system, test_queries):
    metrics = {
        'latency': [],
        'memory_usage': [],
        'cpu_usage': []
    }
    
    for query in test_queries:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        start_cpu = psutil.cpu_percent()
        
        # ทดสอบการทำงาน
        response = rag_system.query(query)
        
        # บันทึกผล
        metrics['latency'].append(time.time() - start_time)
        metrics['memory_usage'].append(
            psutil.Process().memory_info().rss - start_memory
        )
        metrics['cpu_usage'].append(
            psutil.cpu_percent() - start_cpu
        )
    
    # สรุปผล
    return {
        'avg_latency': np.mean(metrics['latency']),
        'avg_memory': np.mean(metrics['memory_usage']),
        'avg_cpu': np.mean(metrics['cpu_usage'])
    }
```

## 5. แนวทางการแก้ไขปัญหา

### 5.1 ปัญหาที่พบบ่อยและวิธีแก้ไข
1. ความแม่นยำต่ำ
   - ปรับค่า chunk_size
   - เพิ่มจำนวน top_k
   - ปรับปรุงคุณภาพเอกสาร

2. ประสิทธิภาพช้า
   - ใช้ระบบ caching
   - ลดขนาด batch
   - ปรับการจัดการหน่วยความจำ

3. ปัญหาภาษาไทย
   - ใช้โมเดลเฉพาะภาษาไทย
   - ปรับปรุงการตัดคำ
   - เพิ่ม preprocessing

### 5.2 การติดตามและแก้ไขปัญหา
```python
import logging

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MonitoredRAGSystem(ChainOfNoteRAG):
    def query(self, question: str, **kwargs):
        try:
            logging.info(f"Processing query: {question}")
            response = super().query(question, **kwargs)
            logging.info("Query processed successfully")
            return response
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise
```

## 6. กรณีศึกษา

### 6.1 ระบบช่วยเหลือลูกค้า
```python
class CustomerSupportRAG:
    def __init__(self):
        self.rag_system = ChainOfNoteRAG(
            embedding_model=MistralEmbeddings()
        )
        self.load_support_documents()
    
    def load_support_documents(self):
        # โหลดเอกสารสำหรับช่วยเหลือลูกค้า
        loader = DocumentLoader()
        loader.load_directory("support_docs/")
        self.rag_system.process_documents_from_loader(loader)
    
    def handle_customer_query(self, query: str):
        # ประมวลผลคำถามลูกค้า
        response = self.rag_system.query(
            query,
            top_k=5,
            return_context=True
        )
        
        # จัดรูปแบบคำตอบ
        return {
            'answer': response['answer'],
            'relevant_docs': [
                doc['metadata']['topic']
                for doc in response['context']
            ],
            'confidence': response.get('confidence', 0.0)
        }
```

### 6.2 ระบบวิเคราะห์เอกสารภายใน
```python
class DocumentAnalyzer:
    def __init__(self):
        self.rag_system = ChainOfNoteRAG(
            embedding_model=SentenceTransformerEmbeddings()
        )
    
    def analyze_documents(self, documents: List[dict]):
        # เพิ่มเอกสารเข้าระบบ
        self.rag_system.add_documents(documents)
        
        # คำถามวิเคราะห์
        analysis_questions = [
            "ประเด็นหลักที่พบในเอกสารมีอะไรบ้าง?",
            "มีความเชื่อมโยงระหว่างเอกสารอย่างไร?",
            "ข้อเสนอแนะที่พบในเอกสารมีอะไรบ้าง?"
        ]
        
        # วิเคราะห์
        analysis_results = {}
        for question in analysis_questions:
            response = self.rag_system.query(question)
            analysis_results[question] = response['answer']
        
        return analysis_results
```

ตัวอย่างเหล่านี้แสดงให้เห็นวิธีการใช้งานระบบในสถานการณ์จริง และวิธีการปรับแต่งระบบให้เหมาะกับความต้องการเฉพาะ ผู้พัฒนาสามารถใช้เป็นแนวทางในการพัฒนาต่อยอดระบบของตนเองได้