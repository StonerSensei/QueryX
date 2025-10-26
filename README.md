# QueryX — Natural Language to SQL Query Generation System

## Project Details

**Course Code and Title:** Major Project (PR1107)  
**Project Supervisor:** Dr. Devika Kataria  

**Team Members:**
- Parth Tiwari (2022btech069)  
- Harshit Dan (2022btech039)  
- Vibhor Saxena (2022btech110)

---

## Abstract

QueryX is an intelligent system designed to enable users to interact with relational databases using natural language instead of SQL.  
It aims to reduce the technical barrier for non-technical users such as business analysts and managers by translating English queries into executable SQL commands.  
The system integrates a Retrieval-Augmented Generation (RAG) pipeline and Large Language Models (LLMs) to generate context-aware SQL queries based on database schema information.

---

## Problem Statement

Accessing and interacting with relational databases such as PostgreSQL typically requires SQL knowledge.  
Most non-technical users are unfamiliar with SQL syntax and prefer to query in natural language (e.g., “Show me the top 5 customers by revenue in the last month”).  
Although AI models can generate SQL queries, they often fail due to a lack of schema awareness, resulting in invalid or inaccurate outputs.  
QueryX addresses this issue by grounding LLMs with actual database schema information through a retrieval-based approach.

**Key Issues Identified:**
1. Lack of schema awareness in traditional LLM-generated queries.  
2. Accuracy and hallucination issues in SQL generation.  
3. Accessibility gap for non-technical users.

---

## Objectives

1. To design and develop a system that allows users to query relational databases using natural language instead of SQL.  
2. To implement a RAG pipeline for schema retrieval and context-aware query generation.  
3. To integrate a Large Language Model (LLM) for generating SQL queries based on retrieved schema context.  
4. To develop a secure backend using Spring Boot for handling database operations and authentication.  
5. To execute generated SQL queries on a PostgreSQL database and return results in both raw and formatted forms.

---

## Scope of the Project

- Utilize a vector database for efficient schema storage and retrieval.  
- Ensure schema-aware query generation using a retrieval-augmented approach.  
- Provide a secure backend interface through Spring Boot with JWT authentication.  
- Enable execution and validation of SQL queries on relational databases.

---

## System Design and Methodology

1. **User Query Input**  
   The user submits a natural language query through a web-based frontend.

2. **Schema Storage and Retrieval**  
   Database schema (table names, columns, relationships) is converted into embeddings and stored in a vector database such as Qdrant or Weaviate.  
   Upon receiving a query, the system retrieves the most relevant schema context.

3. **SQL Generation**  
   The retrieved schema context is provided to a Large Language Model (e.g., OpenAI GPT or Meta’s LLaMA) to generate a valid SQL query.

4. **Query Execution**  
   The generated SQL is executed securely on PostgreSQL, and the results are returned to the user interface.

---

## System Architecture

The architecture consists of the following major components:

- **Frontend:** React with Tailwind CSS  
- **Backend:** Spring Boot (Java)  
- **Database:** PostgreSQL  
- **Vector Database:** Qdrant / Weaviate  
- **Embeddings:** Sentence Embedding Models  
- **Authentication:** JWT (JSON Web Token)  
- **Containerization:** Docker

---

## Future Enhancements

1. Multi-turn conversation handling with context retention.  
2. Predictive analytics and dashboard integration for business intelligence.  
3. Support for multiple databases (MySQL, Oracle, NoSQL).  
4. Improved caching and response optimization.

---

## References

1. Rajkumar, S., & Irsoy, O. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.  
2. Chen, M., Tworek, J., Jun, H., et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374.  
3. Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. arXiv:2005.11401.  
4. PostgreSQL Documentation. [https://www.postgresql.org/docs/](https://www.postgresql.org/docs/)  
5. Qdrant Vector Database Documentation. [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)  
6. OpenAI API Documentation. [https://platform.openai.com/docs](https://platform.openai.com/docs)  
7. Spring Boot Reference Guide. [https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/](https://docs.spring.io/spring-boot/docs/current/reference/htmlsingle/)  
8. JWT.io. *Introduction to JSON Web Tokens*. [https://jwt.io/introduction](https://jwt.io/introduction)  
9. Hugging Face Sentence Transformers. [https://www.sbert.net](https://www.sbert.net)  
10. Weaviate Vector Search Engine. [https://weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)

---

## Repository Structure
<img width="978" height="589" alt="image" src="https://github.com/user-attachments/assets/43448bdb-cc80-4f84-b4da-58a20222a593" />
<img width="1001" height="591" alt="image" src="https://github.com/user-attachments/assets/37f37db5-1f25-405f-8e19-e501a90576ab" />


