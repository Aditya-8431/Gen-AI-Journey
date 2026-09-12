from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""
Maximum attention: output, errors, 
loops, arrays, strings, searching and 
sorting.
2 — HIGH Computer Science + DSA
Focus on the specific topics listed 
above, especially arrays, strings, 
searching and sorting.
3 — HIGH SQL + OOPS + JavaScript
SQL: GROUP BY, HAVING, 
aggregate functions, JOINs and 
RANK. JavaScript: DOM, event 
listeners and functions. OOPS: core 
fundamentals.
4 — HIGH Networking + Cyber Security + 
DevOps + Cloud
Prepare the fundamentals of these 
four areas; the transcript identifies 
them as the difficult portion of the syllabus.
"""

splitter= RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunk=splitter.split_text(text)

print(chunk)
print(len(chunk))
