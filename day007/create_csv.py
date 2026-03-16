import pandas as pd

content = {
    "students": ["Rahul", "Prince", "Kapil", "Gautam", "Mayank"],
    "marks": [80, 89, 98, 95, 78],
    "age": [18, 19, 17, 17, 20]
}
pd.DataFrame(content).to_csv("./testfile.csv", index=False)