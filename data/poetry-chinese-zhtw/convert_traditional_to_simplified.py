from opencc import OpenCC

cc = OpenCC('t2s')  # Traditional -> Simplified

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

converted = cc.convert(text)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(converted)
