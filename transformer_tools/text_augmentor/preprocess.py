import re

def sub_placeholder(string, mask=" "):
    return re.sub(r"{[a-zA-Z1-9\s]*}",mask, string)

def map_apostrophe(string):
# rule based
    mapping = {"'s":" es",
              "'nem":" einem",
               "'ne":" eine",
               "'ner":" einer",
               "'nen'": " einen",
               "'n": " ein"}
    for key in mapping:
        string = re.sub(key, mapping[key],string)
    return string

def preprocess(text):
    try:
        out = text.lower()
        out = sub_placeholder(out)
        out = map_apostrophe(out)
        out = re.sub(" +", " ", out)
    except:
        print("failed in processing text")
        out = ""
    return out