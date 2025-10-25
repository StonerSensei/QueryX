from sqlalchemy import create_engine, inspect
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
from qdrant_client.http.models import VectorParams, Distance

engine = create_engine("postgresql://postgres:mypassword123@localhost:5432/ragDB")

inspector = inspect(engine)

def table_to_description(table_name, inspector):
    cols = inspector.get_columns(table_name)
    pks = inspector.get_pk_constraint(table_name).get("constrained_columns", [])
    fks = inspector.get_foreign_keys(table_name)

    desc = f"Table {table_name}: stores information about {table_name}. "
    col_descs = []
    for col in cols:
        name, type_ = col["name"], str(col["type"])
        role = []
        if name in pks:
            role.append("PK")
        for fk in fks:
            if name in fk["constrained_columns"]:
                role.append(f"FK → {fk['referred_table']}.{fk['referred_columns'][0]}")
        role_str = f" ({', '.join(role)})" if role else ""
        col_descs.append(f"{name} {type_}{role_str}")
    desc += "Columns are " + ", ".join(col_descs) + "."
    return desc

schema_texts = []
for table in inspector.get_table_names():
    schema_texts.append(table_to_description(table, inspector))

print("\n".join(schema_texts))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
embeddings = model.encode(schema_texts).tolist()

qdrant = QdrantClient("localhost", port=6333)

collection_name = "schema_descriptions"

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
)


qdrant.upload_points(
    collection_name="schema_descriptions",
    points=[
        models.PointStruct(id=i, vector=embeddings[i], payload={"description": schema_texts[i]})
        for i in range(len(schema_texts))
    ],
)

print("\nSchema descriptions embedded and stored in Qdrant!")


