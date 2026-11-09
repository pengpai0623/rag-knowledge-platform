from typing import List
from langchain_core.documents import Document
from pptx import Presentation


class PptxParser:
    @staticmethod
    def parse(file_path: str) -> List[Document]:
        docs = []
        prs = Presentation(file_path)

        for slide_idx, slide in enumerate(prs.slides, 1):
            # 提取页面所有文本
            text_fragments = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            text_fragments.append(text)

            # 提取备注
            note_text = ""
            if slide.has_notes_slide:
                note_text = slide.notes_slide.notes_text_frame.text.strip()

            full_text = "\n".join(text_fragments)
            if note_text:
                full_text += f"\n\n演讲备注：\n{note_text}"

            if not full_text.strip():
                continue

            docs.append(Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "file_type": "pptx",
                    "position": f"第{slide_idx}页",
                    "slide_index": slide_idx,
                    "has_note": bool(note_text)
                }
            ))

        return docs

if __name__ == "__main__":
    file_Path = "LCEL与RetrievalQA对比.pptx"
    docus = PptxParser.parse(file_path=file_Path)
    print(f"documents数量：{len(docus)}")
    for i in docus:
        print(f"读取到的page_content: {i.page_content[:60000]}", end="\n")