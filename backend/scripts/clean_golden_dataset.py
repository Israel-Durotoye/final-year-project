import re
import os

file_path = "c:/Users/inspi/OneDrive/Desktop/FINAL_YEAR_PROJECT/backend/data/knowledge_base/golden_dataset.md"
backup_path = "c:/Users/inspi/OneDrive/Desktop/FINAL_YEAR_PROJECT/backend/data/knowledge_base/golden_dataset.md.bak"

def clean_dataset():
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Save a backup just in case
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Original size: {len(content)} bytes")
    
    # We will split into paragraphs and filter them
    paragraphs = content.split('\n\n')
    cleaned_paragraphs = []
    
    # Patterns to match conversational boilerplate
    bad_patterns = [
        r"^(Here are|Here is|These are).*(extracted|key points|summary|facts).*:?\s*$",
        r"^(Based on the provided text|Based on the text).*(summarize|answer).*:?\s*$",
        r"^(I will summarize|I'll provide a summary).*",
        r".*You didn't ask any specific questions.*",
        r".*Please let me know if you have any specific questions.*",
        r".*If you would like me to continue summarizing.*",
        r".*It appears that the text has been cut off.*",
        r"^(The text|This chapter|The chapter|The section|This section) (discusses|describes|highlights|explains|provides|focuses on|also notes|also mentions|concludes by noting).*",
        r"^(The main points discussed are|Some key takeaways from the text include|Key points from the text include|Here are the main points|Overall, the text|Overall, this chapter|Therefore, the final answer|In summary).*",
        r"^(Here's a summary|Summary:|Key Points:|Questions:|Here are some key points).*",
    ]
    
    # Compile regexes
    regexes = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in bad_patterns]
    
    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue
            
        # Check if paragraph matches any bad pattern
        is_bad = False
        
        # If it's a very short conversational sentence or matches explicitly
        for regex in regexes:
            if regex.match(p_stripped):
                is_bad = True
                break
                
        # Also let's do a line-by-line check for mixed paragraphs
        if not is_bad:
            lines = p_stripped.split('\n')
            good_lines = []
            for line in lines:
                line_bad = False
                for regex in regexes:
                    if regex.match(line.strip()):
                        line_bad = True
                        break
                if not line_bad:
                    good_lines.append(line)
            
            if good_lines:
                cleaned_paragraphs.append('\n'.join(good_lines))
        
    # Rejoin with double newlines
    cleaned_content = '\n\n'.join(cleaned_paragraphs)
    
    # Fix triple+ newlines if any
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
        
    print(f"Cleaned size: {len(cleaned_content)} bytes")

if __name__ == "__main__":
    clean_dataset()
