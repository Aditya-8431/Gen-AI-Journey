from pydantic import BaseModel,EmailStr
from typing import Optional

class Student(BaseModel):
    name:str ='Aditya'
    Age: Optional[int] =None
    email: EmailStr



new_student={'Age':32,'email':'abc@gmail.com'};

student=Student(**new_student)

print(student)