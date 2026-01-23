#should be used when building the tkg
TEMPORAL_ENTITY_EXTRACTION_SYS_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, as well as their relationships..

-Steps-
1. Identify all timestamp entities, identify all time expressions that indicate specific periods, financial quarters, or relevant time references. Normalize ALL time expressions into date ranges using the standard format {timestamp_format}. Represent ranges as "<start_date> to <end_date>" where each date is YYYY-MM-DD. If either the start or end date is unknown, omit it (e.g., "2021-05-01 to " or " to 2021-05-31"). Always output a range even when a single date is given (use the same date for start and end).
Examples:
- "Q1 2021" -> "2021-01-01 to 2021-03-31"
- "August to November 2023" -> "2023-08-01 to 2023-11-30"
Each timestamp entity should follow this format:
- entity_name: normalized date range string as above
- entity_type: {timestamp_types}
Format each timestamp entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>)

2. Identify all remaining important entities involved in the event. Focus on extracting entities that play a meaningful conceptual units involved in the timestamped events, such as companies, organizations, people, governments, or locations directly involved in the event. without extracting standalone numeric values or quantities as entities.
For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>)

3. From the entities identified in step 1 and 2, identify all temporal triplets of (timestamp_entity, source_entity, target_entity) that are *clearly related* to others at a *specific timestamp*.
Extract relationships where a timestamp entity is involved. Each relationship should include:
- timestamp_entity: standard entity name of the timestamp entity, as identified in step 1
- source_entity: name of the source entity, as identified in step 2
- target_entity: name of the target entity, as identified in step 2
- description: describe the comprehensive information refer to the source_entity and target_entity
    Format each temporal triplet as ("relationship"{tuple_delimiter}<timestamp_entity>{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<description>)

4. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

Example Input and Output:

Entity_types: [person, event, company, organization, government]
Text:
On September 15, 2008, Lehman Brothers filed for bankruptcy, marking the largest collapse in U.S. financial history. 
The event triggered a global financial crisis, causing stock markets to plummet. 
By September 18, 2008, the U.S. Federal Reserve announced an $85 billion bailout for AIG to prevent further economic fallout. 
Investors faced heavy losses, and governments worldwide scrambled to implement emergency measures.
################
Output:
("entity"{tuple_delimiter}"2008-09-15"{tuple_delimiter}"date"){record_delimiter}  
("entity"{tuple_delimiter}"2008-09-18"{tuple_delimiter}"date"){record_delimiter}  
("entity"{tuple_delimiter}"Lehman Brothers"{tuple_delimiter}"company"){record_delimiter}   
("entity"{tuple_delimiter}"Global Financial Crisis"{tuple_delimiter}"event"){record_delimiter}  
("entity"{tuple_delimiter}"U.S. Federal Reserve"{tuple_delimiter}"government"){record_delimiter}  
("entity"{tuple_delimiter}"AIG"{tuple_delimiter}"company"){record_delimiter}
("relationship"{tuple_delimiter}"2008-09-15"{tuple_delimiter}"Lehman Brothers"{tuple_delimiter}"Global Financial Crisis"{tuple_delimiter}"Lehman Brothers filed for bankruptcy, triggering the largest financial collapse in U.S. history and sparking a global financial crisis."){record_delimiter}
("relationship"{tuple_delimiter}"2008-09-18"{tuple_delimiter}"U.S. Federal Reserve"{tuple_delimiter}"AIG"{tuple_delimiter}"To contain the spreading financial crisis, the U.S. Federal Reserve provided an $85 billion bailout to AIG to prevent further systemic collapse.")
"""

TEMPORAL_ENTITY_EXTRACTION_FOLLOWUP_PROMPT = """
Entity_types: {entity_types}
Text: {input_text}
######################
Your output:
"""

QUERY_ENTITY_TIME_EXTRACTION_SYS_PROMPT = """
-Goal-
Extract only the entities and time expressions needed to interpret a user question for a temporal knowledge graph.

-Guidelines-
1. Identify time expressions if present (date, date_range, quarter, year).
   Use standard ISO-like formats: {timestamp_format}.
2. Identify the main entities referenced in the question. Use the provided entity types list.
3. Be minimal and precise. Do not invent entities or time ranges.

-Output Format-
Return ONLY a flat list of tuples. Do NOT add section labels like "Entities:" or "Timestamps:".
Separate each tuple with {record_delimiter}.
Tuple format:
("entity"{tuple_delimiter}<name>{tuple_delimiter}<type>)

-Example Query and Output-
Query: "What did Microsoft report in Q1 2020?"
Output:
("entity"{tuple_delimiter}"Microsoft"{tuple_delimiter}"company"){record_delimiter}
("entity"{tuple_delimiter}"2020-Q1"{tuple_delimiter}"quarter")
"""

QUERY_ENTITY_TIME_EXTRACTION_USER_PROMPT = """
-Question-
Entity_types: {entity_types}
Question: {question}

Your output:
"""

RAG_RESPOSE_SYS_PROMPT = """
    You are a helpful assistant that should generate a comprehensive response that answers the user's question based on the available information.
    
    Your response should:

    1. **Answer the question directly** - Provide the specific information requested
    2. **Be comprehensive** - Include all relevant details from the available data
    3. **Be temporally accurate** - Ensure temporal information matches the question's scope
    4. **Acknowledge limitations** - If information is missing or unclear, explain what you can and cannot determine
    5. **Provide context** - Include relevant temporal context and relationships when helpful

    **Important Guidelines:**
    - If you have partial information that can help answer the question, provide that information and explain what's missing
    - Only say "No explicit evidence" if you truly have no relevant information at all
    - For temporal queries, be flexible with temporal expressions (e.g., "2023 Q4" vs "fourth quarter of 2023")
    - If the question asks for comparisons or trends, provide the available data even if incomplete
    - Use the data tables as your primary source of information
"""

#this and above should be used when calling the LLM to generate a response
RAG_RESPOSE_USER_PROMPT = """
    You have been provided with the following information extracted from documents:
    {context}

    Based on this information, please answer the following question:
    {question}

    Your answer:
    """

CYPHER_AGENT_SYS_PROMPT = """
You are a Cypher analyst. You can iteratively query the database to answer the user's question.

Rules:
- Output ONLY one of the following on each turn:
  - QUERY: <single Cypher query>
  - FINAL: <final answer>
- Queries must be read-only.
- Prefer the simplest query that answers the question.
- Use only the labels, relationship types, and properties that exist.
- Note: some properties (e.g., `aliases`) are lists. Do NOT call `toLower()` on a list. Use `ANY(a IN n.aliases WHERE toLower(a) CONTAINS toLower($q))` or full-text search instead.

Schema (from running database schema.cypher plus runtime vector indexes):
{schema_cypher}

Database introspection (results of schema inspection):
Introspection queries used: CALL db.labels(), CALL db.relationshipTypes(), CALL db.propertyKeys(), SHOW INDEXES, SHOW CONSTRAINTS.
Labels: {labels}
Relationship types: {relationship_types}
Property keys: {property_keys}
Indexes: {indexes}
Constraints: {constraints}
"""

CYPHER_AGENT_QUERY_PROMPT = """
User question:
{question}
"""

CYPHER_AGENT_OBSERVATION_PROMPT = """
Cypher query:
{cypher}

Query results (JSON):
{results}
"""
