# เอกสารอ้างอิง API (ฉบับภาษาไทย)

เอกสารนี้ให้ภาพรวมโดยย่อของคลาสและฟังก์ชันหลักในระบบ Chain-of-Note RAG สำหรับข้อมูลโดยละเอียด โปรดดูที่ซอร์สโค้ดในไดเรกทอรี `src/`

## `src/rag_system.py`

### `class ChainOfNoteRAG`

คลาสหลักที่จัดการกระบวนการ RAG ทั้งหมด

*   **`__init__(self, embedding_model: Optional[EmbeddingModel] = None, llm_model_name: str = "google/flan-t5-large")`**: เริ่มต้นระบบ `ChainOfNoteRAG`
    *   `embedding_model`:  อินสแตนซ์ของ `EmbeddingModel` (ไม่บังคับ, ค่าเริ่มต้นคือ `SentenceTransformerEmbeddings`)
    *   `llm_model_name`: ชื่อของ Language Model ที่จะใช้สำหรับการสร้าง Note และคำตอบ (ค่าเริ่มต้นคือ "google/flan-t5-large")

*   **`add_documents(self, documents: List[Dict]) -> None`**: เพิ่มรายการเอกสารในระบบ แต่ละเอกสารควรเป็น Dictionary ที่มีคีย์ "content" เป็นอย่างน้อย

*   **`process_documents_from_loader(self, loader: DocumentLoader, chunk_size: int = 500) -> None`**: ประมวลผลเอกสารจากอินสแตนซ์ `DocumentLoader`

*   **`query(self, query: str, top_k: int = 5, return_context: bool = False, return_notes: bool = False) -> Dict[str, Any]`**: ประมวลผลคำถามของผู้ใช้และส่งคืน Dictionary ที่มีคำตอบ และตัวเลือกสำหรับ Note ที่สร้างขึ้นและเอกสารบริบทที่ดึงมา

## `src/chain_of_note.py`

### `class ChainOfNote`

คลาสนี้จัดการตรรกะหลักของ Chain-of-Note: การสร้าง Note ขั้นกลางและการสังเคราะห์คำตอบสุดท้าย

*   **`__init__(self, model_name: str = "google/flan-t5-large")`**: เริ่มต้นอินสแตนซ์ `ChainOfNote`
    *   `model_name`: ชื่อของ Language Model ที่จะใช้

*   **`generate_note(self, query: str, documents: List[Dict]) -> str`**: สร้าง Note ขั้นกลางตามคำถามและเอกสารที่ดึงมา

*   **`generate_answer(self, query: str, notes: str, documents: List[Dict]) -> str`**: สร้างคำตอบสุดท้ายตามคำถาม, Note และเอกสารที่ดึงมา

*   **`generate_response(self, query: str, documents: List[Dict], return_notes: bool = False) -> Dict[str, Any]`**: สร้างการตอบสนองที่สมบูรณ์ รวมถึง Note และคำตอบสุดท้าย

## `src/embeddings.py`

### `class EmbeddingModel`

คลาสฐานนามธรรมสำหรับ Embedding Models

*   **`embed_documents(self, documents: List[str]) -> np.ndarray`**:  Embed รายการข้อความเอกสาร
*   **`embed_query(self, query: str) -> np.ndarray`**: Embed คำถามของผู้ใช้

### `class SentenceTransformerEmbeddings(EmbeddingModel)`

การใช้งาน `EmbeddingModel` โดยใช้ Sentence Transformers

*    **`__init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2")`**: เริ่มต้นด้วยโมเดล Sentence Transformer ที่เฉพาะเจาะจง

### `class HuggingFaceEmbeddings(EmbeddingModel)`
การใช้งานโดยใช้ Hugging Face Transformers

## `src/document_store.py`

### `class DocumentStore`

จัดการการจัดเก็บและการดึงข้อมูลเอกสารและ Embeddings

*   **`__init__(self, embedding_dim: int = 768)`**: เริ่มต้น Document Store
*   **`add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None`**: เพิ่มเอกสารและ Embeddings ลงใน Store
*   **`search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]`**: ค้นหาเอกสาร k อันดับแรกที่คล้ายกับ Embedding คำถามมากที่สุด
*  **`save(self, file_path: str) -> None`**: บันทึก Document Store ลงในไฟล์
*  **`load(self, file_path: str) -> None`**: โหลด Document Store จากไฟล์

## `src/data_loader.py`

### `class DocumentLoader`

จัดการการโหลดและการประมวลผลเอกสารล่วงหน้าจากแหล่งต่างๆ

*   **`__init__(self)`**: เริ่มต้น `DocumentLoader`
*   **`load_text(self, text: str, metadata: Optional[Dict] = None) -> None`**: โหลดเอกสารข้อความเดียว
*   **`load_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None`**: โหลดเอกสารข้อความหลายรายการ
*   **`load_csv(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]] = None) -> None`**: โหลดเอกสารจากไฟล์ CSV
*   **`create_chunks(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]`**: แบ่งเอกสารที่โหลดออกเป็นส่วนย่อยๆ
*   **`get_dataset(self) -> Dataset`**: คืนค่าเอกสาร

## `src/advanced_techniques.py`
### `class EnhancedChainOfNote(ChainOfNote)`
* **`__init__(self, model_name: str = "google/flan-t5-large", verification_model_name: str = "google/flan-t5-base")`**: เริ่มต้น EnhancedChainOfNote
* **`generate_response(self, query: str, documents: List[Dict], return_notes: bool = False, return_verification: bool = False) -> Dict[str, Any]`**: สร้างการตอบสนองต่อคำถาม พร้อมด้วยบันทึกขั้นกลางและการตรวจสอบการอ้างสิทธิ์ (เป็นทางเลือก)

## `src/evaluation.py`
### `class RAGEvaluator`
*   **`__init__(self)`**: เริ่มต้น RAGEvaluator
*   **`evaluate_response(self, query: str, generated_answer: str, reference_answer: str, retrieved_docs: List[Dict]) -> Dict[str, Any]`**: ประเมินคำตอบที่สร้างขึ้นเดียวกับคำตอบอ้างอิงและเอกสารที่ดึงมา
*   **`compare_systems(self, queries: List[str], ground_truths: List[str], system_a_responses: List[Dict], system_b_responses: List[Dict]) -> Dict[str, Any]`**: เปรียบเทียบประสิทธิภาพของระบบ RAG สองระบบ (เช่น RAG มาตรฐานเทียบกับ Chain-of-Note RAG) ในชุดคำถาม