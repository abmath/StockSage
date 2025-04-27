import markdown
import os
from weasyprint import HTML, CSS
import io

def convert_md_to_pdf(input_file, output_file):
    """Convert a markdown file to PDF using WeasyPrint"""
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Read the markdown file
    with open(input_file, 'r') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Add basic styling to the HTML
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 2cm;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #3498db;
                margin-top: 30px;
            }}
            h3 {{
                color: #5D76A9;
            }}
            code {{
                background-color: #f8f8f8;
                padding: 2px 5px;
                border-radius: 3px;
                font-family: monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
                border: 1px solid #ddd;
            }}
            .toc {{
                background-color: #f8f8f8;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            .toc ul {{
                list-style-type: none;
                padding-left: 20px;
            }}
            @page {{
                margin: 2cm;
                @top-center {{
                    content: "Stock Portfolio Analyzer with AI Advisors - User Manual";
                    font-size: 9pt;
                    color: #666;
                }}
                @bottom-center {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                }}
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF using WeasyPrint
    HTML(string=html_with_style).write_pdf(output_file)
    
    print(f"PDF generated successfully: {output_file}")

if __name__ == "__main__":
    input_file = "user_manual.md"
    output_file = "Stock_Portfolio_Analyzer_User_Manual.pdf"
    
    if os.path.exists(input_file):
        convert_md_to_pdf(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")