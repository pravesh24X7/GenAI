from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "No Name"   # default value
    age: int = 18   # this is the default value
    email: Optional[EmailStr]
    cgpa: float = Field(gt=0, lt=10, default=5, description="Decimal value representing the CGPA of student.")    # Field is used to set constraints, description works similar to Annotation in TypedDict

new_student = {
    "name": "pravesh",
    "age": str(23),     # here we passed as string, but it converted as int datatype, which we've specified in Student class. this is called implicit data conversion (in python this is referred as type coresing)
    "email": "kiituso.2885@gmail.com",   # if we don't pass a valid email address, it throws an error.
}

prvsh = Student(**new_student)
print(prvsh)

print(type(prvsh.age), type(prvsh.name))

print(dict(prvsh))  # for converting it into a dictionary.
print(prvsh.model_dump_json())  # for converting it into a json object.