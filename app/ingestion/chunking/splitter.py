#Long text is split into small paragraph based chunks of 1500 characters
#to be processed and embedded for RAG appln

import logfire

# Split text into smaller chunks based on paragraphs
def chunk_text(text: str, chunk_size: int = 1500) -> list[str]:
    """
    Splits text by paragraphs and keeps chunks within the specified size.
    """
    # Track the text chunking process
    with logfire.span("✂️ Text Chunking", text_length=len(text)):

        # Return an empty list if the text is empty
        if not text.strip():
            return []

        # Split text wherever there are two newlines
        paragraphs = text.split("\n\n")

        # Store the generated chunks
        chunks = []

        # Hold paragraphs until the chunk reaches the size limit
        current_chunk = ""

        # Process each paragraph
        for p in paragraphs:

            # Add the paragraph if the chunk stays within the size limit
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"

            else:
                # Save the current chunk if it contains text
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start a new chunk with the current paragraph
                current_chunk = p + "\n\n"

        # Save the final chunk if it contains text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Remove any empty chunks
        valid_chunks = [c for c in chunks if c.strip()]

        # Log the total number of chunks created
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")

        # Return the list of valid chunks
        return valid_chunks