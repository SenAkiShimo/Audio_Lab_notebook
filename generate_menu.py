import os
import re

NOTES_DIR = "notes"
INDEX_FILE = "index.html"

def extract_html_info(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            

        title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else os.path.basename(file_path).replace('.html', '')

        pure_text = re.sub(r'<[^>]+>', ' ', content)
        pure_text = re.sub(r'\s+', ' ', pure_text).strip()
        
        # 截取前 300 个字作为搜索索引
        keywords = pure_text[:300].replace('"', '&quot;') 
        
        return title, keywords
    except Exception as e:
        filename = os.path.basename(file_path).replace('.html', '')
        return filename, filename

def generate_html_menu():
    if not os.path.exists(NOTES_DIR):
        os.makedirs(NOTES_DIR, exist_ok=True)
        print(f"已自动为你创建【{NOTES_DIR}】总文件夹，请往里面丢你的分类文件夹和笔记网页。")
        return None

    html_snippets = []
    is_first_link = True
    default_src = ""
    

    categories = sorted([d for d in os.listdir(NOTES_DIR) if os.path.isdir(os.path.join(NOTES_DIR, d)) and not d.startswith('.')])
    
    for category in categories:
        cat_path = os.path.join(NOTES_DIR, category)
        
        html_snippets.append('            <div class="category-item">')
        html_snippets.append(f'                <div class="category-title">📂 {category}</div>')
        html_snippets.append('                <ul class="nav-links">')
        
        files = sorted([f for f in os.listdir(cat_path) if f.endswith('.html') and not f.startswith('.')])
        
        for file in files:
            file_relative_path = f"{NOTES_DIR}/{category}/{file}"
            title, keywords = extract_html_info(os.path.join(cat_path, file))
            
            if is_first_link:
                html_snippets.append(f'                    <li><a href="{file_relative_path}" target="content-frame" data-keywords="{keywords}" class="active">{title}</a></li>')
                default_src = file_relative_path
                is_first_link = False
            else:
                html_snippets.append(f'                    <li><a href="{file_relative_path}" target="content-frame" data-keywords="{keywords}">{title}</a></li>')
                
        html_snippets.append('                </ul>')
        html_snippets.append('            </div>')
        
    return "\n".join(html_snippets), default_src

def update_index_html():
    result = generate_html_menu()
    if not result: return
    menu_content, default_src = result
    
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        content = f.read()
        
    start_tag = "<!-- MENU_START -->"
    end_tag = "<!-- MENU_END -->"
    
    if start_tag in content and end_tag in content:
        before_menu = content.split(start_tag)[0]
        after_menu = content.split(end_tag)[1]
        
        new_content = f"{before_menu}{start_tag}\n{menu_content}\n            {end_tag}{after_menu}"
        
        if default_src:
            new_content = re.sub(r'src="[^"]*"\s+frameborder="0"', f'src="{default_src}" frameborder="0"', new_content)
            
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("自动化同步成功")
    else:
        print("未在 index.html 中找到需要的 <!-- MENU_START --> 标记。")

if __name__ == "__main__":
    update_index_html()
