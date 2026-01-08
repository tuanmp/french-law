import os
from uuid import uuid4

import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_PATH=os.environ['PSCRATCH']

metadata_cols = [
    "article_identifier",
    "article_num",
    'article_date_debut',
    'texte_nature',
    'texte_titre',
]
text_col = "article_contenu_text"

MODEL = "OrdalieTech/Solon-embeddings-large-0.1"
# MODEL = "intfloat/multilingual-e5-large-instruct" 

CHUNK_SIZE=300
OVERLAP=30
BATCH_SIZE=2048

def main():


    print("Initializing model and tokenizer")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=OVERLAP, 
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n\n\n", "\n\n", "\n", " ", ""]
    )

    print("Getting data ...")
    data = pd.read_csv("/global/cfs/cdirs/m3443/usr/pmtuan/learn_hf_agent/combined-data.csv", chunksize=20000)

    print("Initiating ChromaDB")
    chroma_client = chromadb.PersistentClient(path=os.path.join(DB_PATH, "french_law"))
    # chroma_client = chromadb.EphemeralClient()
    chroma_batch_size = chroma_client.get_max_batch_size()
    chroma_batch_size = min(BATCH_SIZE, chroma_batch_size)


    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL,
        model_kwargs={'device': "cuda"},
        encode_kwargs={"batch_size": BATCH_SIZE},
        query_encode_kwargs={"batch_size": BATCH_SIZE},
        show_progress=True
    )

    vector_store = Chroma(
        client=chroma_client,
        collection_name="laws_v1",
        embedding_function=embeddings,
    )

    # collection = chroma_client.get_or_create_collection(name="laws_v1")

    print("Starting feature extraction ...")

    for df_chunk in data:

        documents = []
        df_chunk.dropna(inplace=True)
        

        for idx, row in df_chunk.iterrows():
            if not row[text_col]: continue

            metadata = {col: row[col] for col in metadata_cols if col in row}
            documents.append(
                Document(page_content=row[text_col], metadata=metadata)
            )

        if len(documents) == 0: continue
        documents = splitter.split_documents(documents)
            
        uuids = [str(uuid4()) for _ in range(len(documents))]
        # print(documents)
        for batch, ids in [(documents[i:i + chroma_batch_size], uuids[i: i+chroma_batch_size]) for i in range(0, len(documents), chroma_batch_size)]:
            vector_store.add_documents(documents=batch, ids=ids)
        
    pass 

if __name__=="__main__":

    main()


