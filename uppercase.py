word = input("Enter a word: ")
result = ""

for ch in word:
    
    if ch >= 'a' and ch <= 'z':
        result += chr(ord(ch) - 32)   
    else:
        result += ch                  

print(result)
