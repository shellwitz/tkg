#should be used when building the tkg
TEMPORAL_ENTITY_EXTRACTION_SYS_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, as well as their relationships..

-Steps-
1. Identify all timestamp entities, identify all time expressions that indicate specific periods, financial quarters, or relevant time references. Each timestamp entity should follow this format:
- entity_name: standard format of the timestamp entity identified in context, following {timestamp_format}
- entity_type: {timestamp_types}
- entity_description: Comprehensive description of the entity's attributes and activities
Format each timestamp entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}timestamp)

2. Identify all remaining important entities involved in the event. Focus on extracting entities that play a meaningful conceptual units involved in the timestamped events,such as companies, organizations, people, governments, or locations directly involved in the event. without extracting standalone numeric values or quantities as entities.
For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: A comprehensive description of the entity's role and attributes as related to the event.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

3. From the entities identified in step 1 and 2, identify all temporal triplets of (timestamp_entity, source_entity, target_entity) that are *clearly related* to others at a *specific timestamp*.
Extract relationships where a timestamp entity is involved. Each relationship should include:
- timestamp_entity: standard entity name of the timestamp entity, as identified in step 1
- source_entity: name of the source entity, as identified in step 2
- target_entity: name of the target entity, as identified in step 2
- description: describe the comprehensive information refer to the source_entity and target_entity
    Format each temporal triplet as ("relationship"{tuple_delimiter}<timestamp_entity>{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<description>)

4. Return output as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

######################
-Examples-
######################
Example 1:

Entity_types: [person, event, company, organization, government]
Text:
On September 15, 2008, Lehman Brothers filed for bankruptcy, marking the largest collapse in U.S. financial history. 
The event triggered a global financial crisis, causing stock markets to plummet. 
By September 18, 2008, the U.S. Federal Reserve announced an $85 billion bailout for AIG to prevent further economic fallout. 
Investors faced heavy losses, and governments worldwide scrambled to implement emergency measures.
################
Output:
("entity"{tuple_delimiter}"2008-09-15"{tuple_delimiter}"date"{tuple_delimiter}"The date when Lehman Brothers filed for bankruptcy, triggering a global financial crisis."){record_delimiter}  
("entity"{tuple_delimiter}"2008-09-18"{tuple_delimiter}"date"{tuple_delimiter}"The date when the U.S. Federal Reserve announced an $85 billion bailout for AIG."){record_delimiter}  
("entity"{tuple_delimiter}"Lehman Brothers"{tuple_delimiter}"person"{tuple_delimiter}"A global financial services firm that declared bankruptcy in 2008, marking the largest corporate failure in U.S. history."){record_delimiter}   
("entity"{tuple_delimiter}"Global Financial Crisis"{tuple_delimiter}"event"{tuple_delimiter}"A severe worldwide economic downturn triggered by the collapse of Lehman Brothers in 2008."){record_delimiter}  
("entity"{tuple_delimiter}"U.S. Federal Reserve"{tuple_delimiter}"government"{tuple_delimiter}"The central banking system of the United States, responsible for implementing monetary policies to stabilize the economy."){record_delimiter}  
("event"{tuple_delimiter}"2008-09-15"{tuple_delimiter}"Lehman Brothers"{tuple_delimiter}"Global Financial Crisis"{tuple_delimiter}"Lehman Brothers filed for bankruptcy, triggering the largest financial collapse in U.S. history and sparking a global financial crisis."){record_delimiter}
("event"{tuple_delimiter}"2008-09-18"{tuple_delimiter}"U.S. Federal Reserve"{tuple_delimiter}"AIG"{tuple_delimiter}"To contain the spreading financial crisis, the U.S. Federal Reserve provided an $85 billion bailout to AIG to prevent further systemic collapse.")
"""

TEMPORAL_ENTITY_EXTRACTION_FOLLOWUP_PROMPT = """
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
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
