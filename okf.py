import os
import requests
import google.generativeai as genai

# ==========================================
# CONFIGURATION
# ==========================================
CONFLUENCE_DOMAIN = "your-domain.atlassian.net"
CONFLUENCE_EMAIL = "your-email@company.com"
CONFLUENCE_API_TOKEN = "your-confluence-api-token"
PAGE_ID = "123456789"
ATTACHMENT_ID = "987654321" # The ID of the Draw.io export (PNG/JPG)

GEMINI_API_KEY = "your-gemini-api-key"
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_confluence_page(page_id):
    """Fetches the title and raw HTML body of a Confluence page."""
    url = f"https://{CONFLUENCE_DOMAIN}/wiki/rest/api/content/{page_id}?expand=body.storage"
    auth = (CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)
    headers = {"Accept": "application/json"}
    
    response = requests.get(url, auth=auth, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    return data['title'], data['body']['storage']['value']

def download_confluence_attachment(page_id, attachment_id):
    """Downloads the Draw.io PNG attachment from Confluence."""
    url = f"https://{CONFLUENCE_DOMAIN}/wiki/rest/api/content/{page_id}/child/attachment"
    auth = (CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN)
    
    response = requests.get(url, auth=auth, headers={"Accept": "application/json"})
    response.raise_for_status()
    
    attachments = response.json().get('results', [])
    download_path = next((item['_links']['download'] for item in attachments if item['id'] == f"att{attachment_id}"), None)
    
    if not download_path:
        raise ValueError("Attachment not found.")
        
    download_url = f"https://{CONFLUENCE_DOMAIN}/wiki{download_path}"
    img_response = requests.get(download_url, auth=auth)
    img_response.raise_for_status()
    
    temp_filename = "temp_diagram.png"
    with open(temp_filename, 'wb') as f:
        f.write(img_response.content)
        
    return temp_filename

def synthesize_okf_document(title, html_content, image_path):
    """Uses Gemini 1.5 Pro to process HTML and Image into a unified OKF document."""
    print("Uploading diagram to Gemini...")
    vision_file = genai.upload_file(path=image_path)
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    You are an expert system architect migrating a Confluence page titled "{title}" to Google's Open Knowledge Format (OKF).
    
    I am providing you with two things:
    1. The raw HTML content of the Confluence page.
    2. An image of the architecture diagram attached to the page.
    
    Your task is to synthesize this into a single, highly readable OKF Markdown document.
    
    STRICT REQUIREMENTS:
    - Start the document with valid YAML frontmatter enclosing it in `---`. Include fields for `type`, `title`, `description`, `tags`, and `status`.
    - Extract all valuable technical knowledge from the HTML. Strip out messy Atlassian macros, navigation metadata, and irrelevant layout tags.
    - Translate the provided image diagram into accurate Mermaid.js syntax. Embed this Mermaid codeblock logically within the text where the architecture is being discussed.
    - Organize the text using clear Markdown headings, bullet points, and tables where appropriate.
    - Output ONLY the raw Markdown text. Do not wrap your response in ```markdown tags.
    
    Raw HTML Content:
    {html_content}
    """
    
    print("Generating OKF document...")
    response = model.generate_content([prompt, vision_file])
    
    genai.delete_file(vision_file.name)
    
    # Clean up output in case the LLM wrapped it in markdown code blocks anyway
    final_text = response.text.strip()
    if final_text.startswith("```markdown"):
        final_text = final_text[11:]
    if final_text.endswith("```"):
        final_text = final_text[:-3]
        
    return final_text.strip()

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        print("1. Fetching Confluence Page HTML...")
        title, html_body = get_confluence_page(PAGE_ID)
        
        print("2. Downloading Diagram Attachment...")
        image_path = download_confluence_attachment(PAGE_ID, ATTACHMENT_ID)
        
        print("3. Synthesizing OKF via Multimodal LLM (This may take a minute)...")
        final_okf = synthesize_okf_document(title, html_body, image_path)
        
        # Save to file
        output_filename = f"{title.replace(' ', '_').lower()}.md"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_okf)
            
        if os.path.exists(image_path):
            os.remove(image_path)
            
        print(f"\nSuccess! Multimodal OKF file generated: {output_filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
