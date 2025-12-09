import json
from typing import List
from pathlib import Path

# Unstructured for document parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# LangChain components
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
#######################################################################


def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res",  # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text. If you want table to extracted as images, add "Table" to extract_image_block_types below
        extract_image_block_types=[
            "Image"
        ],  # Grab only images found in the PDF as image blocks
        # extract_image_block_types tells partition_pdf what types of blocks to extract
        # as "image" blocks when using strategy="hi_res". You can set any element type you
        # want here, such as "Figure", "Table", "Formula", etc.
        extract_image_block_to_payload=True,  # Store images as base64-encoded data you can actually use
        # extract_image_block_output_dir =    # If you want to save images to disk, specify a directory here. It will trigger
        # only when extract_image_block_to_payload is Flase
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements


#########################################################################
def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements,  # The parsed PDF elements from previous step
        max_characters=3000,  # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400,  # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500,  # Merge tiny chunks under 500 chars with neighbors
    )

    print(f"✅ Created {len(chunks)} chunks")
    return chunks


############################################################################
def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        "text": chunk.text,  # because text is always present in a chunk as we useed title-based chunking
        "tables": [],
        "images": [],
        "types": ["text"],
        "source_file": (
            chunk.metadata.filename if hasattr(chunk.metadata, "filename") else None
        ),
        "page_number": (
            chunk.metadata.page_number
            if hasattr(chunk.metadata, "page_number")
            else None
        ),
    }  # we will loop through chunks original elements and complete the above dict
    # we seperat images and tables, because want to turrn them into some kind of "text" (summarize) so that when quary the RAG system, it will have a better undestanting of these elements. We didnt do such thing for fourmula as it is aleady text-basd

    # Check for tables and images in original elements
    if hasattr(chunk, "metadata") and hasattr(
        chunk.metadata, "orig_elements"
    ):  # This check seems unnecessary but safe as i think the chunk will alwasys have metadata and orig_elements (Not sure!)
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(
                    element.metadata, "text_as_html", element.text
                )  # It will get table as HTML (getting the 'text_as_html' attribute)
                # if infer_table_structure=True (in partitioning step),
                # else If HTML is not available, it falls back to using the raw text (element.text)
                # , which are the text extracted from table ( jumbled text, not clean HTML)
                content_data["tables"].append(table_html)

            # Handle images
            elif element_type == "Image":
                if hasattr(element, "metadata") and hasattr(
                    element.metadata, "image_base64"
                ):
                    content_data["types"].append("image")
                    content_data["images"].append(element.metadata.image_base64)

    content_data["types"] = list(
        set(content_data["types"])
    )  # It removes duplicate values from the list of types.
    return content_data


#####################################################################
# the function creates AI-enhanced summary for chunks that contain tables and/or images
def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Build the text prompt
        prompt_text = f"""You are an expert system designed to create highly searchable document
        representations for Retrieval-Augmented Generation (RAG).

        Below is the content extracted from a PDF. Your job is to convert ALL of 
        this into a unified, richly descriptive, highly searchable text block.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

        prompt_text += """
        YOUR TASK:
        Generate a single, comprehensive **search-optimized description** of the content.
        Combine insights from text, tables, and images.

        Your description must include:

        1. Key facts, numbers, and data points from text and tables  
        2. Main topics, concepts, and themes present in the content  
        3. Questions that a user could answer using this content  
        4. Observations from the images (they follow below)  
        5. Alternative search terms / synonyms users might use  
        6. Anything that improves future retrieval quality

        Write in a factual, information-dense style. 
        Do NOT summarize; instead, **expand** the content into a searchable representation.

        ------------------------
        SEARCHABLE DESCRIPTION (OUTPUT BELOW)
        ------------------------
        """

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for image_base64 in images:
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary


def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        # print(f"   Processing chunk {current_chunk}/{total_chunks}")

        # Analyze chunk content
        content_data = separate_content_types(chunk)

        # Debug prints
        # print(f"     Types found: {content_data['types']}")
        # print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Create AI-enhanced summary if chunk has tables/images
        if content_data["tables"] or content_data["images"]:
            # print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data["text"], content_data["tables"], content_data["images"]
                )
                # print(f"     → AI summary created successfully")
                # print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data["text"]
        else:
            # print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data["text"]

        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": content_data["text"],
                        "tables_html": content_data["tables"],
                        "images_base64": content_data["images"],
                        "source_file": content_data["source_file"],
                        "page_number": content_data["page_number"],
                    }
                )  # for now, we dumps as json string, but later we convert it back to dictionay by using jason.load
            },
        )

        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


# ----------------------------------------------------
# Main ingestion pipeline for a single file
# ----------------------------------------------------

# partition_document(file_path)  -> returns elements
# create_chunks_by_title(elements) -> returns chunks
# summarise_chunks(chunks) -> returns List[Document]


def process_single_pdf(file_path: str) -> List[Document]:
    print(f"\n📄 Processing file: {file_path}")

    # Step 1: Partition
    elements = partition_document(file_path)

    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)

    # Step 3: AI Summaries
    summarised_docs = summarise_chunks(chunks)

    print(f"✔️  Created {len(summarised_docs)} summarised chunks")
    return summarised_docs


# ----------------------------------------------------
# Build the vectorstore (Chroma)
# ----------------------------------------------------
def build_vectorstore(path_folder: str, persist_dir: str = "chroma_db"):

    # path_folder = "./docs"  # Folder containing PDF files
    # persist_dir = "db/chroma_db_2"  # Directory to persist Chroma DB
    pdf_folder = Path(path_folder)

    if not pdf_folder.exists():
        raise ValueError(f"Folder does not exist: {pdf_folder}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create/Load Chroma DB
    vectorstore = Chroma(
        collection_name="rag_ingestion",
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

    all_docs = []

    # Loop over PDF files
    pdf_files = list(pdf_folder.glob("*.pdf"))
    print(f"🔍 Found {len(pdf_files)} PDF files\n")

    for pdf_path in pdf_files:
        try:
            docs = process_single_pdf(str(pdf_path))
            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            continue

    # Add documents to vectorstore
    if all_docs:
        print(f"\n📥 Adding {len(all_docs)} documents to Chroma...")
        vectorstore.add_documents(all_docs)

        print("💾 Vector store persisted to disk.")

    print("\n🎉 Ingestion completed.")


if __name__ == "__main__":

    build_vectorstore(path_folder="./docs", persist_dir="db/chroma_db_2")
