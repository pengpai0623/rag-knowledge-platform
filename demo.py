from utils import UniversalDocParser
from utils import OCRPDFParser

if __name__ == "__main__":
    file_Path = r"docs\rag\UN38.3.pdf"
    # docus = UniversalDocParser().parse(file_Path)
    docus = OCRPDFParser().parse_by_page(file_Path)
    print(f"document数量：{len(docus)}")
    for i in docus:
        print(f"读取到的page_content: {i.page_content[:600]}", end="\n")