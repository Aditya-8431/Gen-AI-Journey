from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""

grade = "A"
elif marks >= 75:
grade = "B"
else:
grade = "C"
print(grade)
 For Loop for i in range(1, 6):
print(i)
 While Loop i = 1
while i <= 5:
print(i)
i += 1
 def add(a, b):
return a + b
result = add(10, 20)
print(result)

def greet(name="User"):
print("Hello", name)
greet()
greet("Aditya")
 [10, 20, 30, 40]
nums.append(50)
nums.remove(20)
print(nums)
print(max(nums), min(nums))
 {
"name": "Aditya",
"age": 21,
"course": "CSE"
}
print(student["name"])
student["age"] = 22

nums.add(5)
print(nums)
print(2 in nums)"""

splitter= RecursiveCharacterTextSplitter.from_language(
    language="python",
    chunk_size=100,
    chunk_overlap=0
)

chunks=splitter.split_text(text)
print(chunks[1])