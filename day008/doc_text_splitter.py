from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text="""
    class Student:
        def __init__(self, name, age, marks):
            self.age = age
            self.name = name
            self.marks = marks
        
        def __repr__(self):
            return self.name
        
        def is_passing(self):
            return self.marks >= 60
    
    rahul = Student(name="Rahul", age=19, marks=71)
    print(rahul)

    if rahul.is_passing():
        print("Rahul passed")
    else:
        print("Rahul failed")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

results = splitter.split_text(text=text)
print(results)