# DoubleRAG

DoubleRAG is built on two main layers and a shared hierarchical database. The highest in the hierarchy is "topic," which describe the major topic of a piece of text, then "children" which describes the subtopic, and then the texts themselves.

## Ingestion Layer
The first one is an ingestion layer. When the user wants to give a document for the RAG agent to use in the future. The ingestion layer chunks the text in a file, then for each chunk, it looks for the overarching topic, then it finds a subtopic, and then it looks at the documents within the subtopic.
If there is sufficient overlap, the agent will attempt to "merge" with an exists document in order to limit redundancy. If at any point, the model finds that the chunk does not fit into a certain topic/subtopic/file, it will create a new topic/subtopic/file and place it there.

## RAG Layer
The navigation of the RAG layer is very similar to the ingestion layer, but instead of modifying the database, it just pulls documents relevant to the query and inserts them with the user prompt.
